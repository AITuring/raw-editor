//! Supported spatial affine correction of a native owner's low-frequency band.
//! Inputs are immutable native fields and ownership; no source decoding or I/O.
#![allow(dead_code)]

use super::tone::{self, PairField, ToneTile};

#[derive(Clone, Debug)]
pub(crate) struct LocalToneField {
    pub(crate) width: usize,
    pub(crate) height: usize,
    pub(crate) stride: u32,
    /// Per tile/cell: three gain increments followed by three offsets.
    pub(crate) parameters: Vec<Vec<[f32; 6]>>,
    pub(crate) fitted_relations: Vec<(usize, usize)>,
    pub(crate) iterations: usize,
    pub(crate) retained_samples: Vec<usize>,
    pub(crate) maximum_parameter_change: f32,
}

struct Observation {
    cell: usize,
    weight: [f32; 3],
    first: [f32; 3],
    second: [f32; 3],
}

struct Edge {
    first: usize,
    second: usize,
    observations: Vec<Observation>,
}

impl LocalToneField {
    pub(crate) fn at(&self, tile: usize, x: f64, y: f64) -> [f32; 6] {
        let Some(field) = self.parameters.get(tile) else {
            return [0.0; 6];
        };
        let stride = f64::from(self.stride);
        let u = (x - (stride - 1.0) * 0.5) / stride;
        let v = (y - (stride - 1.0) * 0.5) / stride;
        let (u0, v0) = (u.floor(), v.floor());
        let mut sum = [0.0; 6];
        let mut total = 0.0;
        for (dy, wy) in [(0.0, 1.0 - (v - v0)), (1.0, v - v0)] {
            for (dx, wx) in [(0.0, 1.0 - (u - u0)), (1.0, u - u0)] {
                let (cx, cy) = (u0 + dx, v0 + dy);
                if cx < 0.0 || cy < 0.0 || cx >= self.width as f64 || cy >= self.height as f64 {
                    continue;
                }
                let weight = (wx * wy) as f32;
                let value = field[cy as usize * self.width + cx as usize];
                for channel in 0..6 {
                    sum[channel] += value[channel] * weight
                }
                total += weight;
            }
        }
        if total > 1e-9 {
            for value in &mut sum {
                *value /= total
            }
        }
        sum
    }

    pub(crate) fn corrected(&self, tile: usize, low: [f32; 3], x: f64, y: f64) -> [f32; 3] {
        let p = highlight_supported_parameters(self.at(tile, x, y), low);
        std::array::from_fn(|c| low[c] * (1.0 + p[c]) + p[c + 3])
    }
}

pub(crate) fn highlight_supported_parameters(mut p: [f32; 6], low: [f32; 3]) -> [f32; 6] {
    // Channels above the fitting range carry no ordinary calibration evidence.
    // Extrapolating a neighbouring affine fit into clipped white can darken one
    // source while clipping the other and manufacture a chromatic seam. Return
    // smoothly to the native channel at saturation, using only its sigma=64
    // low band; the original high-frequency component remains untouched.
    for c in 0..3 {
        let strength = ((1.0 - low[c]) / (1.0 - tone::TONE_MAX_SAMPLE_LUMINANCE)).clamp(0.0, 1.0);
        p[c] *= strength;
        p[c + 3] *= strength;
    }
    p
}

fn usable(value: f32) -> bool {
    (tone::TONE_MIN_SAMPLE_LUMINANCE..=tone::TONE_MAX_SAMPLE_LUMINANCE).contains(&value)
}
fn luminance(rgb: [f32; 3]) -> f32 {
    rgb[0] * 0.2126 + rgb[1] * 0.7152 + rgb[2] * 0.0722
}

pub(crate) fn fit_local_tone(
    tiles: &[ToneTile],
    relations: &[(usize, usize)],
    owners: &[u16],
    output_width: u32,
    output_height: u32,
) -> Option<LocalToneField> {
    fit_local_tone_with_identity_prior(tiles, relations, owners, output_width, output_height, 0.06)
}

pub(crate) fn fit_local_tone_with_identity_prior(
    tiles: &[ToneTile],
    relations: &[(usize, usize)],
    owners: &[u16],
    output_width: u32,
    output_height: u32,
    identity_prior: f32,
) -> Option<LocalToneField> {
    if tiles.is_empty()
        || output_width == 0
        || output_height == 0
        || owners.len() != output_width as usize * output_height as usize
    {
        return None;
    }
    let stride = tone::TONE_GRID_DIVISOR;
    let width = output_width.div_ceil(stride) as usize;
    let height = output_height.div_ceil(stride) as usize;
    let count = width.checked_mul(height)?;
    // A bounded analysis allocation rather than a second full-resolution RGB canvas.
    if count.checked_mul(tiles.len())? > 40_000_000 {
        return None;
    }
    let position = |cell: usize| {
        (
            (cell % width) as f64 * f64::from(stride) + f64::from(stride - 1) * 0.5,
            (cell / width) as f64 * f64::from(stride) + f64::from(stride - 1) * 0.5,
        )
    };
    let sigma = 64.0 / f64::from(stride);
    let radius = (3.0 * sigma).ceil() as isize;
    let kernel_sum = (-radius..=radius)
        .map(|offset| (-(offset as f64).powi(2) / (2.0 * sigma * sigma)).exp())
        .sum::<f64>();
    let normalizer = kernel_sum * kernel_sum;
    // Preserve every native ownership pixel in the analysis mask. A single
    // centre probe loses narrow valid remnants at a RAW coverage boundary.
    let mut owner_index =
        vec![None; tiles.iter().map(|t| t.owner_id).max().unwrap_or(0) as usize + 1];
    for (index, tile) in tiles.iter().enumerate() {
        owner_index[tile.owner_id as usize] = Some(index);
    }
    let mut counts = vec![vec![0u16; count]; tiles.len()];
    for (pixel, &owner) in owners.iter().enumerate() {
        let Some(index) = owner_index.get(owner as usize).copied().flatten() else {
            continue;
        };
        let cell = (pixel / output_width as usize / stride as usize) * width
            + pixel % output_width as usize / stride as usize;
        counts[index][cell] += 1;
    }
    let mut weights = Vec::with_capacity(tiles.len());
    for native_counts in counts {
        let mask = native_counts
            .into_iter()
            .map(|n| [f64::from(n) / f64::from(stride * stride)])
            .collect::<Vec<_>>();
        let blurred = tone::gaussian_cells(&mask, width, height, 64.0 / f64::from(stride));
        // Fixed normalisation preserves zero padding at the canvas boundary.
        weights.push(
            blurred
                .iter()
                .map(|a| (a[0] / normalizer) as f32)
                .collect::<Vec<_>>(),
        );
    }
    let mut unique = relations.to_vec();
    let pair_boundaries = native_boundary_cells(owners, output_width, output_height, stride);
    let boundary_strength = strength_from_boundary_cells(
        width,
        height,
        stride,
        pair_boundaries.values().flatten().copied(),
    );
    // Boundary activity depends on output ownership, while calibration
    // connectivity depends on actual validated overlap coverage. Keeping those
    // separate preserves identity in owner interiors without removing covered
    // intermediary stations from a boundary's calibration graph.
    let activity = (0..count)
        .map(|cell| {
            let sum = weights.iter().map(|w| w[cell]).sum::<f32>();
            let squares = weights.iter().map(|w| w[cell] * w[cell]).sum::<f32>();
            ((sum * sum - squares) * 0.5).max(0.0)
        })
        .collect::<Vec<_>>();
    unique.retain(|&(a, b)| a != b && a < tiles.len() && b < tiles.len());
    for pair in &mut unique {
        if pair.0 > pair.1 {
            *pair = (pair.1, pair.0)
        }
    }
    unique.sort_unstable();
    unique.dedup();
    let mut edges = Vec::new();
    let mut fitted_relations = Vec::new();
    let mut retained_samples = vec![0; tiles.len()];
    for (first, second) in unique {
        let Some(field) = PairField::new(&tiles[first], &tiles[second]) else {
            continue;
        };
        // Two low-pass sigmas separate coherent local illumination changes
        // from isolated corrupt measurements without changing the 3xMAD rule.
        let (consistency, retained) = field.spatial_consistency(128.0);
        if retained < tone::TONE_MIN_SAMPLES {
            continue;
        }
        retained_samples[first] = retained_samples[first].max(retained);
        retained_samples[second] = retained_samples[second].max(retained);
        let mut observations = Vec::new();
        let key = (
            tiles[first].owner_id.min(tiles[second].owner_id),
            tiles[first].owner_id.max(tiles[second].owner_id),
        );
        let direct_strength = strength_from_boundary_cells(
            width,
            height,
            stride,
            pair_boundaries.get(&key).into_iter().flatten().copied(),
        );
        for cell in 0..count {
            // A validated overlap remains evidence when only one station owns
            // nearby output. Requiring both to own pixels removes intermediary
            // stations from the local calibration graph and leaves thin owner
            // remnants without an indirect, validated calibration path.
            // A real narrow seam carries the same pairwise obligation as a
            // broad seam. Its data weight cannot depend on the smaller owner's
            // area. Only validated relations receive this direct seam weight.
            let base = direct_strength[cell]
                + weights[first][cell] * weights[second][cell]
                + 0.1 * activity[cell].max(boundary_strength[cell]);
            if base <= 1e-7 {
                continue;
            }
            let (x, y) = position(cell);
            let Some((a, b, sample_x, sample_y)) =
                supported_analysis_cell_sample(&field, x, y, stride)
            else {
                continue;
            };
            let (la, lb) = (luminance(a), luminance(b));
            let ordinary = usable(la) && usable(lb);
            if ordinary {
                let Some(key) = field.spatial_window_key(sample_x, sample_y, 128.0) else {
                    continue;
                };
                if !consistency.get(&key).is_some_and(|model| {
                    model.accepts(&tone::ToneSample {
                        owner: a,
                        source: b,
                    })
                }) {
                    continue;
                }
            }
            let highlight =
                la > tone::TONE_MAX_SAMPLE_LUMINANCE || lb > tone::TONE_MAX_SAMPLE_LUMINANCE;
            let weight = std::array::from_fn(|c| {
                if ordinary || (highlight && usable(a[c]) && usable(b[c])) {
                    base
                } else {
                    0.0
                }
            });
            if weight.iter().all(|v| *v == 0.0) {
                continue;
            }
            observations.push(Observation {
                cell,
                weight,
                first: a,
                second: b,
            });
        }
        fitted_relations.push((first, second));
        edges.push(Edge {
            first,
            second,
            observations,
        });
    }
    if edges.is_empty() {
        return None;
    }
    let mut neighbours = vec![Vec::new(); tiles.len()];
    for (index, edge) in edges.iter().enumerate() {
        neighbours[edge.first].push((index, true));
        neighbours[edge.second].push((index, false));
    }
    let mut normal = Vec::with_capacity(tiles.len());
    let mut degrees = Vec::with_capacity(tiles.len());
    for (index, tile) in tiles.iter().enumerate() {
        let mut rows = (0..count)
            .map(|cell| {
                let (x, y) = position(cell);
                let low = tone::sample_tile_bilinear(tile, x, y).unwrap_or([0.0; 3]);
                // Owned interiors must retain their native contrast. Keep the
                // direct seam obligation strong, while increasing the identity
                // prior against corrections extrapolated through weak overlap.
                let anchor = identity_prior * weights[index][cell];
                std::array::from_fn::<_, 9, _>(|k| match k / 3 {
                    0 => anchor * low[k % 3] * low[k % 3] + 1e-5,
                    1 => anchor * low[k % 3],
                    _ => anchor + 1e-3,
                })
            })
            .collect::<Vec<_>>();
        let mut degree = vec![[0.0f32; 3]; count];
        for &(edge_index, first) in &neighbours[index] {
            for observation in &edges[edge_index].observations {
                let rgb = if first {
                    observation.first
                } else {
                    observation.second
                };
                let row = &mut rows[observation.cell];
                for c in 0..3 {
                    let w = observation.weight[c];
                    row[c] += w * rgb[c] * rgb[c];
                    row[c + 3] += w * rgb[c];
                    row[c + 6] += w;
                    degree[observation.cell][c] += w;
                }
            }
        }
        normal.push(rows);
        degrees.push(degree);
    }
    let mut parameters = vec![vec![[0.0f32; 6]; count]; tiles.len()];
    // Cells without an observation have a zero RHS and a positive identity
    // prior, so their unique minimizer stays exactly zero in every sweep.
    let active_cells = degrees
        .iter()
        .map(|tile| {
            tile.iter()
                .enumerate()
                .filter_map(|(cell, d)| d.iter().any(|v| *v > 0.0).then_some(cell))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let mut rhs = vec![[0.0f32; 6]; count];
    let mut iterations = 0;
    let mut maximum_parameter_change = f32::INFINITY;
    for iteration in 0..1000 {
        let mut maximum = 0.0f32;
        for tile in 0..tiles.len() {
            for &cell in &active_cells[tile] {
                rhs[cell] = [0.0; 6];
            }
            for &(edge_index, first) in &neighbours[tile] {
                let edge = &edges[edge_index];
                let other = if first { edge.second } else { edge.first };
                for observation in &edge.observations {
                    let (a, b) = if first {
                        (observation.first, observation.second)
                    } else {
                        (observation.second, observation.first)
                    };
                    let p = parameters[other][observation.cell];
                    for c in 0..3 {
                        let target = b[c] - a[c] + p[c] * b[c] + p[c + 3];
                        rhs[observation.cell][c] += observation.weight[c] * a[c] * target;
                        rhs[observation.cell][c + 3] += observation.weight[c] * target;
                    }
                }
            }
            for &cell in &active_cells[tile] {
                for c in 0..3 {
                    let n = normal[tile][cell];
                    let r = rhs[cell];
                    let (alpha, offset) = tone::bounded_low_frequency_affine_coordinate(
                        [f64::from(n[c]), f64::from(n[c + 3]), f64::from(n[c + 6])],
                        [f64::from(r[c]), f64::from(r[c + 3])],
                    );
                    maximum = maximum
                        .max((parameters[tile][cell][c] - alpha as f32).abs())
                        .max((parameters[tile][cell][c + 3] - offset as f32).abs());
                    parameters[tile][cell][c] = alpha as f32;
                    parameters[tile][cell][c + 3] = offset as f32;
                }
            }
        }
        iterations = iteration + 1;
        maximum_parameter_change = maximum;
        if maximum < 1e-5 {
            break;
        }
    }
    drop(normal);
    drop(edges);
    let mass = (0..count)
        .map(|cell| weights.iter().map(|w| w[cell]).sum::<f32>())
        .collect::<Vec<_>>();
    for tile in 0..tiles.len() {
        extend_supported_parameters(
            &mut parameters[tile],
            &degrees[tile],
            &weights[tile],
            &mass,
            &boundary_strength,
            width,
            height,
        );
    }
    Some(LocalToneField {
        width,
        height,
        stride,
        parameters,
        fitted_relations,
        iterations,
        retained_samples,
        maximum_parameter_change,
    })
}

/// Normalised convolution only fills missing model coefficients; no unfit
/// highlight or dark sample becomes a calibration observation.
fn supported_analysis_cell_sample(
    field: &PairField,
    x: f64,
    y: f64,
    stride: u32,
) -> Option<([f32; 3], [f32; 3], f64, f64)> {
    let half = f64::from(stride - 1) * 0.5;
    // A covered boundary pixel may lie in a cell whose centre falls outside
    // the common footprint. Probe only real supported positions in that same
    // analysis cell; never replace missing overlap data with a zero sample.
    for (px, py) in [
        (x, y),
        (x - half, y),
        (x + half, y),
        (x, y - half),
        (x, y + half),
        (x - half, y - half),
        (x + half, y - half),
        (x - half, y + half),
        (x + half, y + half),
    ] {
        if let Some((a, b)) = field.at(px, py) {
            return Some((a, b, px, py));
        }
    }
    None
}

fn extend_supported_parameters(
    parameters: &mut [[f32; 6]],
    degree: &[[f32; 3]],
    owned: &[f32],
    mass: &[f32],
    boundary_strength: &[f32],
    width: usize,
    height: usize,
) {
    let mut missing = degree
        .iter()
        .zip(owned.iter().zip(mass))
        .map(|(d, (own, total))| d.map(|v| v < 1e-7 && total - own > 1e-7))
        .collect::<Vec<_>>();
    for radius in [4, 8, 16, 32, 64] {
        if missing.iter().all(|m| m.iter().all(|v| !*v)) {
            break;
        }
        let support = three_box_blur(
            &degree.iter().map(|v| v.map(f64::from)).collect::<Vec<_>>(),
            width,
            height,
            radius,
        );
        let weighted = parameters
            .iter()
            .zip(degree)
            .map(|(p, d)| std::array::from_fn::<_, 6, _>(|c| f64::from(p[c]) * f64::from(d[c % 3])))
            .collect::<Vec<_>>();
        let values = three_box_blur(&weighted, width, height, radius);
        for cell in 0..parameters.len() {
            for c in 0..3 {
                if missing[cell][c] && support[cell][c] > 1e-6 {
                    let strength = boundary_strength[cell];
                    parameters[cell][c] = ((values[cell][c] / support[cell][c]) as f32 * strength)
                        .clamp(tone::TONE_MIN_GAIN - 1.0, tone::TONE_MAX_GAIN - 1.0);
                    parameters[cell][c + 3] = ((values[cell][c + 3] / support[cell][c]) as f32
                        * strength)
                        .clamp(-tone::TONE_MAX_ABS_OFFSET, tone::TONE_MAX_ABS_OFFSET);
                    missing[cell][c] = false;
                }
            }
        }
    }
}

/// Fade supported model extension with sigma 64 from actual native seams.
/// Unlike ownership area, this does not attenuate a genuine one-pixel seam.
fn native_boundary_strength(owners: &[u16], width: u32, height: u32, stride: u32) -> Vec<f32> {
    let columns = width.div_ceil(stride) as usize;
    let rows = height.div_ceil(stride) as usize;
    let pairs = native_boundary_cells(owners, width, height, stride);
    strength_from_boundary_cells(columns, rows, stride, pairs.values().flatten().copied())
}

fn native_boundary_cells(
    owners: &[u16],
    width: u32,
    height: u32,
    stride: u32,
) -> std::collections::BTreeMap<(u16, u16), Vec<usize>> {
    let columns = width.div_ceil(stride) as usize;
    let mut pairs = std::collections::BTreeMap::<(u16, u16), Vec<usize>>::new();
    let cell = |pixel: usize| {
        (pixel / width as usize / stride as usize) * columns
            + pixel % width as usize / stride as usize
    };
    for (pixel, &owner) in owners.iter().enumerate() {
        if owner == 0 {
            continue;
        }
        for other in [
            (pixel % width as usize + 1 < width as usize).then_some(pixel + 1),
            (pixel / width as usize + 1 < height as usize).then_some(pixel + width as usize),
        ]
        .into_iter()
        .flatten()
        {
            if owners[other] != 0 && owners[other] != owner {
                let list = pairs
                    .entry((owner.min(owners[other]), owner.max(owners[other])))
                    .or_default();
                list.push(cell(pixel));
                list.push(cell(other));
            }
        }
    }
    for list in pairs.values_mut() {
        list.sort_unstable();
        list.dedup();
    }
    pairs
}

fn strength_from_boundary_cells(
    columns: usize,
    rows: usize,
    stride: u32,
    cells: impl Iterator<Item = usize>,
) -> Vec<f32> {
    let mut squared = vec![f64::INFINITY; columns * rows];
    let mut has_boundary = false;
    for cell in cells {
        squared[cell] = 0.0;
        has_boundary = true;
    }
    if !has_boundary {
        return vec![0.0; columns * rows];
    }
    for row in squared.chunks_mut(columns) {
        row.copy_from_slice(&squared_distance_1d(row));
    }
    for x in 0..columns {
        let column = (0..rows)
            .map(|y| squared[y * columns + x])
            .collect::<Vec<_>>();
        let distance = squared_distance_1d(&column);
        for y in 0..rows {
            squared[y * columns + x] = distance[y];
        }
    }
    squared
        .into_iter()
        .map(|d| (-d * f64::from(stride).powi(2) / (2.0 * 64.0 * 64.0)).exp() as f32)
        .collect()
}

/// Exact lower envelope of squared-distance parabolas, including empty rows.
fn squared_distance_1d(input: &[f64]) -> Vec<f64> {
    let mut sites = Vec::<usize>::new();
    let mut starts = Vec::<f64>::new();
    for (q, &value) in input.iter().enumerate() {
        if !value.is_finite() {
            continue;
        }
        let mut start = f64::NEG_INFINITY;
        while let Some(&p) = sites.last() {
            start = (value + (q as f64).powi(2) - input[p] - (p as f64).powi(2))
                / (2.0 * (q - p) as f64);
            if start > *starts.last().unwrap() {
                break;
            }
            sites.pop();
            starts.pop();
        }
        if sites.is_empty() {
            start = f64::NEG_INFINITY;
        }
        sites.push(q);
        starts.push(start);
    }
    if sites.is_empty() {
        return vec![f64::INFINITY; input.len()];
    }
    let mut active = 0;
    (0..input.len())
        .map(|x| {
            while active + 1 < sites.len() && starts[active + 1] <= x as f64 {
                active += 1;
            }
            input[sites[active]] + (x as f64 - sites[active] as f64).powi(2)
        })
        .collect()
}

fn three_box_blur<const N: usize>(
    input: &[[f64; N]],
    width: usize,
    height: usize,
    radius: usize,
) -> Vec<[f64; N]> {
    let mut source = input.to_vec();
    let divisor = (radius * 2 + 1) as f64;
    for _ in 0..3 {
        for horizontal in [true, false] {
            let mut result = vec![[0.0; N]; input.len()];
            let (lines, length) = if horizontal {
                (height, width)
            } else {
                (width, height)
            };
            for line in 0..lines {
                let index = |position: usize| {
                    if horizontal {
                        line * width + position
                    } else {
                        position * width + line
                    }
                };
                let mut sum = [0.0; N];
                for p in 0..radius.min(length.saturating_sub(1)) + 1 {
                    for c in 0..N {
                        sum[c] += source[index(p)][c]
                    }
                }
                for p in 0..length {
                    if p > 0 {
                        if p > radius {
                            for c in 0..N {
                                sum[c] -= source[index(p - radius - 1)][c]
                            }
                        }
                        if p + radius < length {
                            for c in 0..N {
                                sum[c] += source[index(p + radius)][c]
                            }
                        }
                    }
                    result[index(p)] = sum.map(|v| v / divisor);
                }
            }
            source = result;
        }
    }
    source
}

#[cfg(test)]
mod tests {
    #[test]
    fn covered_analysis_cell_edge_is_kept_when_its_centre_has_no_overlap() {
        use image::{GrayImage, Luma, Rgb, Rgb32FImage};
        let tile = |owner, y, value| super::ToneTile {
            station_index: owner as usize,
            owner_id: owner,
            low: Rgb32FImage::from_pixel(64, 64, Rgb([value; 3])),
            cell_mean: Rgb32FImage::from_pixel(64, 64, Rgb([value; 3])),
            validity: GrayImage::from_pixel(64, 64, Luma([255])),
            native_detail: None,
            cell_coverage: vec![1.0; 4096],
            world_origin: (0.0, y),
            world_size: (1024.0, 1024.0),
            world_stride: 16.0,
        };
        let field = super::PairField::new(&tile(1, 7.0, 0.4), &tile(2, 15.0, 0.35)).unwrap();
        assert!(field.at(7.5, 7.5).is_none());
        let (a, b, x, y) = super::supported_analysis_cell_sample(&field, 7.5, 7.5, 16).unwrap();
        assert!((0.0..=15.0).contains(&x) && (0.0..=15.0).contains(&y));
        assert_eq!(field.at(x, y), Some((a, b)));
        assert!((a[0] - 0.4).abs() < 1e-6 && (b[0] - 0.35).abs() < 1e-6);
        assert!(super::supported_analysis_cell_sample(&field, -24.5, -24.5, 16).is_none());
    }
    #[test]
    fn boundary_distance_matches_exact_oracle_and_keeps_a_single_pixel_owner() {
        let values = [f64::INFINITY, 0.0, f64::INFINITY, 2.0, f64::INFINITY, 0.0];
        let measured = super::squared_distance_1d(&values);
        for (x, distance) in measured.iter().enumerate() {
            let expected = values
                .iter()
                .enumerate()
                .map(|(p, v)| *v + (x as f64 - p as f64).powi(2))
                .fold(f64::INFINITY, f64::min);
            assert_eq!(*distance, expected);
        }
        let owners = (0..512 * 32)
            .map(|p| if p % 512 == 255 { 2 } else { 1 })
            .collect::<Vec<_>>();
        let strengths = super::native_boundary_strength(&owners, 512, 32, 16);
        assert_eq!(strengths[15], 1.0);
        assert_eq!(strengths[16], 1.0);
        assert!(strengths[0] < 0.001);
        let empty = super::native_boundary_strength(&vec![1; 512 * 32], 512, 32, 16);
        assert!(empty.iter().all(|v| *v == 0.0));
    }
    use super::*;

    #[test]
    fn native_local_model_preserves_ownership_and_respects_original_bounds() {
        let first = [0.35, 0.4, 0.45];
        let second = first.map(|v| v * 1.1);
        let mut tiles = vec![
            tone::world_aligned_native_tone_tile(0, 1, (0, 0, 511, 511), |_, _| Some(first)),
            tone::world_aligned_native_tone_tile(1, 2, (0, 0, 511, 511), |_, _| Some(second)),
        ];
        let owners = (0..512 * 512)
            .map(|i| if i % 512 < 256 { 1 } else { 2 })
            .collect::<Vec<_>>();
        let before = owners.clone();
        let field = fit_local_tone(&tiles, &[(0, 1)], &owners, 512, 512).unwrap();
        let corrected = (
            field.corrected(0, first, 255.5, 255.5),
            field.corrected(1, second, 255.5, 255.5),
        );
        assert!(tone::delta_e00_rgb(corrected.0, corrected.1) < 1.5);
        assert_eq!(owners, before);
        for parameters in field.parameters.iter().flatten() {
            for channel in 0..3 {
                assert!(
                    (tone::TONE_MIN_GAIN - 1.0..=tone::TONE_MAX_GAIN - 1.0)
                        .contains(&parameters[channel])
                );
                assert!(parameters[channel + 3].abs() <= tone::TONE_MAX_ABS_OFFSET);
            }
        }
        assert!(fit_local_tone(&tiles, &[], &owners, 512, 512).is_none());
        tiles[0] =
            tone::world_aligned_native_tone_tile(0, 1, (0, 0, 511, 511), |_, _| Some([1.0; 3]));
        tiles[1] =
            tone::world_aligned_native_tone_tile(1, 2, (0, 0, 511, 511), |_, _| Some([1.0; 3]));
        assert!(
            fit_local_tone(&tiles, &[(0, 1)], &owners, 512, 512).is_none(),
            "clipped pixels alone cannot establish a calibration model"
        );
    }

    #[test]
    fn supported_extension_filter_preserves_mass_and_has_at_least_64px_sigma() {
        let mut impulse = vec![[0.0]; 64 * 64];
        impulse[32 * 64 + 32] = [1.0];
        let filtered = three_box_blur(&impulse, 64, 64, 4);
        let mass = filtered.iter().map(|v| v[0]).sum::<f64>();
        let variance = filtered
            .iter()
            .enumerate()
            .map(|(i, v)| ((i % 64) as f64 - 32.0).powi(2) * v[0])
            .sum::<f64>()
            / mass;
        assert!((mass - 1.0).abs() < 1e-12);
        assert!((variance - 20.0).abs() < 1e-10);
        assert!(variance.sqrt() * f64::from(tone::TONE_GRID_DIVISOR) >= 64.0);
    }
}
#[test]
fn clipped_highlights_do_not_extrapolate_a_chromatic_neighbour_fit() {
    let a = [0.99999446, 0.99821544, 0.9827084];
    let b = [0.99775684, 0.98692524, 0.95973295];
    let pa = [
        -0.009008787,
        -0.017083365,
        -0.020523522,
        0.0003254342,
        0.0020178787,
        0.000388684,
    ];
    let pb = [
        0.011213822,
        0.018574355,
        0.008035533,
        -0.000190947,
        -0.0012505094,
        0.000327138,
    ];
    let correct = |p, low: [f32; 3]| {
        let p = highlight_supported_parameters(p, low);
        std::array::from_fn(|c| (low[c] * (1.0 + p[c]) + p[c + 3]).clamp(0.0, 1.0))
    };
    assert!(tone::delta_e00_rgb(correct(pa, a), correct(pb, b)) < 0.75);
    assert_eq!(highlight_supported_parameters(pa, [0.5; 3]), pa);
    assert_eq!(highlight_supported_parameters(pa, [1.0; 3]), [0.0; 6]);
    let p = highlight_supported_parameters(pa, [0.99; 3]);
    let delta = 0.99 * p[0] + p[3];
    let bright = 0.993 + delta;
    let dark = 0.987 + delta;
    assert!((bright - dark - 0.006).abs() < 1e-6);
}
