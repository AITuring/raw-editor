/// The samples of one pair that pass the 3×MAD consistency check (需求 9.4).
///
/// The check runs on the per-sample RGB log gain, the quantity the tone
/// relation is estimated from. A sample is excluded when the distance of its
/// log-gain vector from the per-channel median exceeds three times the median
/// distance, so the three channels always keep the same samples.
pub(crate) fn consistent_samples(samples: &[ToneSample]) -> Vec<ToneSample> {
    if samples.is_empty() {
        return Vec::new();
    }
    let gains = samples.iter().map(sample_log_gain).collect::<Vec<_>>();
    let medians: [f64; 3] =
        std::array::from_fn(|channel| median_f64(gains.iter().map(|gain| gain[channel])));
    let deviations = gains
        .iter()
        .map(|gain| {
            (0..3)
                .map(|channel| (gain[channel] - medians[channel]).powi(2))
                .sum::<f64>()
                .sqrt()
        })
        .collect::<Vec<_>>();
    let mad = median_f64(deviations.iter().copied());
    // An exact consensus (MAD 0) keeps the consensus up to rounding.
    let threshold = (3.0 * mad).max(1e-6);
    samples
        .iter()
        .zip(&deviations)
        .filter(|(_, deviation)| **deviation <= threshold)
        .map(|(sample, _)| *sample)
        .collect()
}

/// One accepted pair's multiplicative tone relation. `first` is the owner side
/// and `second` the source side of every retained sample.
struct ToneRelation {
    first: usize,
    second: usize,
    /// Per-channel median of `ln first − ln second` over `samples`.
    log_gain: [f64; 3],
    /// Median RMS distance of the sample log gains from `log_gain`.
    scatter: f64,
    samples: Vec<ToneSample>,
}

impl ToneRelation {
    fn new(first: usize, second: usize, samples: Vec<ToneSample>) -> Self {
        let gains = samples.iter().map(sample_log_gain).collect::<Vec<_>>();
        let log_gain: [f64; 3] =
            std::array::from_fn(|channel| median_f64(gains.iter().map(|gain| gain[channel])));
        let scatter = median_f64(gains.iter().map(|gain| {
            ((0..3)
                .map(|channel| (gain[channel] - log_gain[channel]).powi(2))
                .sum::<f64>()
                / 3.0)
                .sqrt()
        }));
        Self {
            first,
            second,
            log_gain,
            scatter,
            samples,
        }
    }

    /// The joint-solve weight of `photometric.rs`: more retained samples and a
    /// tighter log-gain scatter give a relation more pull.
    fn weight(&self) -> f64 {
        let support = self.samples.len() as f64;
        support
            / (support + TONE_MIN_SAMPLES as f64)
            / (1.0 + (self.scatter / TONE_RELATION_SCATTER_SCALE).powi(2))
    }
}

/// Solve one overlap's tone relation after the 3×MAD consistency check, the
/// two-tile case of the joint solve in [`harmonize_tone_tiles`].
///
/// The gain is the per-channel median log ratio of the retained samples and
/// the offset the robust median of `owner − gain · source` (design, Tone_Harmonizer:
/// "先用 log 增益求解得 gain，再对残差求稳健中位数得 offset"). A median ratio is
/// symmetric and does not flatten towards identity where the two low fields
/// disagree by position, which a least-squares affine fit of one field on the
/// other does. A pair with fewer than [`TONE_MIN_SAMPLES`] retained samples is
/// identity and records the stable `tone_insufficient_samples` identifier.
pub(crate) fn solve_tone_pair(samples: &[ToneSample]) -> ToneSolve {
    let retained = consistent_samples(samples);
    if retained.len() < TONE_MIN_SAMPLES {
        record_insufficient_samples(retained.len());
        return ToneSolve::identity(retained.len());
    }
    let relation = ToneRelation::new(0, 1, retained);
    let solved_gain = relation.log_gain.map(|value| value.exp() as f32);
    let gain =
        std::array::from_fn(|channel| solved_gain[channel].clamp(TONE_MIN_GAIN, TONE_MAX_GAIN));
    let solved_offset: [f32; 3] = std::array::from_fn(|channel| {
        median(
            relation
                .samples
                .iter()
                .map(|sample| sample.owner[channel] - gain[channel] * sample.source[channel]),
        )
    });
    let retained = relation.samples;
    let offset = std::array::from_fn(|channel| {
        solved_offset[channel].clamp(-TONE_MAX_ABS_OFFSET, TONE_MAX_ABS_OFFSET)
    });
    let gain_clamped = solved_gain
        .iter()
        .zip(gain.iter())
        .any(|(solved, bounded)| (*solved - *bounded).abs() > 1e-6)
        || solved_offset
            .iter()
            .zip(offset.iter())
            .any(|(solved, bounded)| (*solved - *bounded).abs() > 1e-6);
    if gain_clamped {
        degradation::record_run_degradation(
            degradation::TONE_GAIN_CLAMPED,
            serde_json::json!({
                "stage": "tone_harmonizer",
                "retained_samples": retained.len(),
                "solved_gain": solved_gain,
                "gain": gain,
                "solved_offset": solved_offset,
                "offset": offset,
            }),
        );
    }
    ToneSolve {
        gain,
        offset,
        solved_gain,
        solved_offset,
        retained_samples: retained.len(),
        gain_clamped,
        status: ToneSolveStatus::Applied,
    }
}

/// Compatibility name for call sites that describe this operation as a tone
/// calibration rather than a pair solve.
pub(crate) fn calibrate_tone_pair(samples: &[ToneSample]) -> ToneSolve {
    solve_tone_pair(samples)
}

fn record_insufficient_samples(retained_samples: usize) {
    degradation::record_run_degradation(
        degradation::TONE_INSUFFICIENT_SAMPLES,
        serde_json::json!({
            "stage": "tone_harmonizer",
            "retained_samples": retained_samples,
            "minimum_samples": TONE_MIN_SAMPLES,
        }),
    );
}

/// Estimate a low-frequency RGB field on a 1/16 grid and bilinearly expand it
/// to the input dimensions.
pub(crate) fn low_frequency_field(image: &Rgb32FImage) -> Rgb32FImage {
    let low = low_frequency_grid(image);
    bilinear_expand(&low, image.width(), image.height())
}

/// Analysis-resolution low-frequency field.  Callers that already retain an
/// ownership grid should keep this image and map it by world coordinates,
/// avoiding a full-resolution intermediate allocation.
pub(crate) fn low_frequency_grid(image: &Rgb32FImage) -> Rgb32FImage {
    if image.width() == 0 || image.height() == 0 {
        return Rgb32FImage::new(image.width(), image.height());
    }
    let low_width = image.width().div_ceil(TONE_GRID_DIVISOR).max(1);
    let low_height = image.height().div_ceil(TONE_GRID_DIVISOR).max(1);
    let low = image::imageops::resize(
        image,
        low_width,
        low_height,
        image::imageops::FilterType::Triangle,
    );
    image::imageops::blur(&low, TONE_LOW_FREQUENCY_SIGMA)
}

/// Separable Gaussian of a `width × height` grid of `N`-channel values with
/// zero padding, so that a normalised convolution sees no value outside it.
fn gaussian_cells<const N: usize>(
    values: &[[f64; N]],
    width: usize,
    height: usize,
    sigma: f64,
) -> Vec<[f64; N]> {
    let radius = (3.0 * sigma).ceil() as isize;
    let kernel = (-radius..=radius)
        .map(|offset| (-(offset as f64).powi(2) / (2.0 * sigma * sigma)).exp())
        .collect::<Vec<_>>();
    let pass = |source: &[[f64; N]], horizontal: bool| {
        let mut output = vec![[0.0; N]; source.len()];
        for y in 0..height {
            for x in 0..width {
                let mut sum = [0.0; N];
                for (tap, weight) in kernel.iter().enumerate() {
                    let offset = tap as isize - radius;
                    let (sx, sy) = if horizontal {
                        (x as isize + offset, y as isize)
                    } else {
                        (x as isize, y as isize + offset)
                    };
                    if sx < 0 || sy < 0 || sx >= width as isize || sy >= height as isize {
                        continue;
                    }
                    let value = source[sy as usize * width + sx as usize];
                    for channel in 0..N {
                        sum[channel] += weight * value[channel];
                    }
                }
                output[y * width + x] = sum;
            }
        }
        output
    };
    pass(&pass(values, true), false)
}

/// Tone_Harmonizer field of one tile on a world-aligned analysis grid.
///
/// `tile` holds the tile's pixels with uncovered pixels exactly zero (the
/// compositor's transparency convention). `bounds` are the tile's inclusive
/// output-pixel bounds `(left, top, right, bottom)` and `to_tile` maps an
/// output pixel to tile pixels with the map the compositor samples through.
///
/// The low-frequency field averages covered tile pixels only (a normalised
/// convolution), so uncovered payload never darkens it, and it is resampled
/// through `to_tile`, so a rotated or projective tile is compared with its
/// neighbours at the right world position. A cell is tone evidence
/// ([`TONE_CELL_SAMPLE`]) only where the tile covers its whole analysis cell
/// (需求 9.1, 9.2, 9.7).
pub(crate) fn world_aligned_tone_tile(
    station_index: usize,
    owner_id: u16,
    tile: &Rgb32FImage,
    bounds: (u32, u32, u32, u32),
    to_tile: impl Fn(f64, f64) -> Option<(f64, f64)>,
) -> ToneTile {
    let divisor = TONE_GRID_DIVISOR.max(1);
    let stride = f64::from(divisor);
    let (left, top, right, bottom) = bounds;
    let world_width = (right.saturating_sub(left) + 1).div_ceil(divisor).max(1);
    let world_height = (bottom.saturating_sub(top) + 1).div_ceil(divisor).max(1);
    let mut low = Rgb32FImage::new(world_width, world_height);
    let mut validity = GrayImage::new(world_width, world_height);
    let mut cell_mean = Rgb32FImage::new(world_width, world_height);
    let mut cell_coverage = vec![0.0f32; world_width as usize * world_height as usize];
    let empty = |low, validity, cell_mean, cell_coverage| ToneTile {
        station_index,
        owner_id,
        low,
        validity,
        world_origin: (f64::from(left), f64::from(top)),
        world_size: (
            f64::from(world_width) * stride,
            f64::from(world_height) * stride,
        ),
        world_stride: stride,
        cell_mean,
        cell_coverage,
    };
    let (tile_width, tile_height) = tile.dimensions();
    if tile_width == 0 || tile_height == 0 {
        return empty(low, validity, cell_mean, cell_coverage);
    }

    // Covered sums, covered counts and cell sizes on the tile's own grid.
    let grid_width = tile_width.div_ceil(divisor) as usize;
    let grid_height = tile_height.div_ceil(divisor) as usize;
    let rows = (0..grid_height)
        .into_par_iter()
        .map(|cell_y| {
            let mut row = vec![([0.0f64; 3], 0u32, 0u32); grid_width];
            let y_start = cell_y as u32 * divisor;
            for y in y_start..(y_start + divisor).min(tile_height) {
                for x in 0..tile_width {
                    let cell = &mut row[(x / divisor) as usize];
                    cell.2 += 1;
                    let pixel = tile.get_pixel(x, y).0;
                    if pixel[0].max(pixel[1]).max(pixel[2]) > 1e-6 {
                        for (sum, value) in cell.0.iter_mut().zip(pixel) {
                            *sum += f64::from(value);
                        }
                        cell.1 += 1;
                    }
                }
            }
            row
        })
        .collect::<Vec<_>>();
    let cells = rows.into_iter().flatten().collect::<Vec<_>>();
    let weights = cells
        .iter()
        .map(|&(_, covered, total)| [f64::from(covered) / f64::from(total.max(1))])
        .collect::<Vec<_>>();
    let weighted = cells
        .iter()
        .zip(&weights)
        .map(|(&(sum, covered, _), weight)| {
            std::array::from_fn(|channel| {
                if covered == 0 {
                    0.0
                } else {
                    sum[channel] / f64::from(covered) * weight[0]
                }
            })
        })
        .collect::<Vec<[f64; 3]>>();
    let sigma = f64::from(TONE_LOW_FREQUENCY_SIGMA);
    let blurred = gaussian_cells(&weighted, grid_width, grid_height, sigma);
    let support = gaussian_cells(&weights, grid_width, grid_height, sigma);
    let field = |cell: usize| -> Option<[f64; 3]> {
        (support[cell][0] > 1e-9)
            .then(|| std::array::from_fn(|channel| blurred[cell][channel] / support[cell][0]))
    };

    for world_y in 0..world_height {
        for world_x in 0..world_width {
            let x = f64::from(left) + f64::from(world_x) * stride + (stride - 1.0) * 0.5;
            let y = f64::from(top) + f64::from(world_y) * stride + (stride - 1.0) * 0.5;
            let Some((tile_x, tile_y)) = to_tile(x, y) else {
                continue;
            };
            if !(tile_x >= 0.0
                && tile_y >= 0.0
                && tile_x < f64::from(tile_width)
                && tile_y < f64::from(tile_height))
            {
                continue;
            }
            let cell_x = ((tile_x / stride) as usize).min(grid_width - 1);
            let cell_y = ((tile_y / stride) as usize).min(grid_height - 1);
            let cell = cell_y * grid_width + cell_x;
            let (cell_sum, covered, total) = cells[cell];
            if covered == 0 {
                continue;
            }
            cell_mean.put_pixel(
                world_x,
                world_y,
                Rgb(cell_sum.map(|channel| (channel / f64::from(covered)) as f32)),
            );
            cell_coverage[world_y as usize * world_width as usize + world_x as usize] =
                covered as f32 / total.max(1) as f32;
            // Bilinear over the covered-only field, renormalised over the
            // neighbours that carry any covered pixel.
            let u = (tile_x - (stride - 1.0) * 0.5) / stride;
            let v = (tile_y - (stride - 1.0) * 0.5) / stride;
            let (u0, v0) = (u.floor(), v.floor());
            let mut sum = [0.0f64; 3];
            let mut total_weight = 0.0f64;
            for (dy, weight_y) in [(0.0, 1.0 - (v - v0)), (1.0, v - v0)] {
                for (dx, weight_x) in [(0.0, 1.0 - (u - u0)), (1.0, u - u0)] {
                    let (nx, ny) = (u0 + dx, v0 + dy);
                    if nx < 0.0 || ny < 0.0 || nx >= grid_width as f64 || ny >= grid_height as f64 {
                        continue;
                    }
                    let neighbour = ny as usize * grid_width + nx as usize;
                    let Some(value) = field(neighbour) else {
                        continue;
                    };
                    let weight = weight_x * weight_y;
                    for channel in 0..3 {
                        sum[channel] += weight * value[channel];
                    }
                    total_weight += weight;
                }
            }
            let value = if total_weight > 1e-12 {
                sum.map(|channel| channel / total_weight)
            } else if let Some(value) = field(cell) {
                value
            } else {
                continue;
            };
            low.put_pixel(world_x, world_y, Rgb(value.map(|channel| channel as f32)));
            let level = if covered == total {
                TONE_CELL_SAMPLE
            } else {
                TONE_CELL_PARTIAL
            };
            validity.put_pixel(world_x, world_y, Luma([level]));
        }
    }
    empty(low, validity, cell_mean, cell_coverage)
}
