fn harmonize_after_ownership_with_evidence(
    panorama: &mut Rgb32FImage,
    owners: &[u16],
    tiles: &[ToneTile],
    relations: &[(usize, usize)],
    world_origin: (f64, f64),
    evidence: Option<&GrayImage>,
) -> ToneReport {
    let mut report = ToneReport::default();
    if panorama.width() == 0 || panorama.height() == 0 || tiles.is_empty() {
        report.status = ToneStatus::Identity;
        record_run_report(report.clone());
        return report;
    }
    let pixel_count = panorama.width() as usize * panorama.height() as usize;
    if owners.len() != pixel_count {
        report.status = ToneStatus::Identity;
        record_run_report(report.clone());
        return report;
    }
    // One relation per accepted Station_Relation, measured on its doubly
    // covered overlap only (需求 9.1). Tiles that are merely placed next to
    // each other without an accepted relation contribute nothing.
    let mut pairs = relations
        .iter()
        .filter(|&&(left, right)| left != right && left < tiles.len() && right < tiles.len())
        .map(|&(left, right)| (left.min(right), left.max(right)))
        .collect::<Vec<_>>();
    pairs.sort_unstable();
    pairs.dedup();
    let mut solved_relations = Vec::with_capacity(pairs.len());
    let mut sample_support = vec![0usize; tiles.len()];
    for (first, second) in pairs {
        let samples = PairField::new(&tiles[first], &tiles[second])
            .map(|field| field.samples())
            .unwrap_or_default();
        let retained = consistent_samples(&samples);
        if retained.len() < TONE_MIN_SAMPLES {
            record_insufficient_samples(retained.len());
            report
                .pairs_without_evidence
                .push(TonePairWithoutEvidenceRecord {
                    left: tiles[first].station_index,
                    right: tiles[second].station_index,
                    retained_samples: retained.len() as u64,
                });
            continue;
        }
        sample_support[first] = sample_support[first].max(retained.len());
        sample_support[second] = sample_support[second].max(retained.len());
        solved_relations.push(ToneRelation::new(first, second, retained));
    }

    // One joint solve over every relation; each transform is bounded once, and
    // the pixels, the boundary re-check and the report read the same bounded
    // values while the report keeps the unbounded solution (需求 9.3, 9.9).
    let transforms = solve_tone_transforms(tiles.len(), &solved_relations);
    for (index, tile) in tiles.iter().enumerate() {
        let transform = &transforms[index];
        if transform.clamped {
            degradation::record_run_degradation(
                degradation::TONE_GAIN_CLAMPED,
                serde_json::json!({
                    "stage": "tone_harmonizer",
                    "station_index": tile.station_index,
                    "solved_gain": transform.solved_gain,
                    "gain": transform.gain,
                    "solved_offset": transform.solved_offset,
                    "offset": transform.offset,
                }),
            );
        }
        report.tiles.push(ToneTileRecord {
            station_index: tile.station_index,
            gain: transform.gain.map(f64::from),
            offset: transform.offset.map(f64::from),
            samples: sample_support[index] as u64,
            clamped: transform.clamped,
            solved_gain: transform.solved_gain.map(f64::from),
            solved_offset: transform.solved_offset.map(f64::from),
        });
    }

    // A tile owns its own high-frequency residual. The owner's low field is
    // expanded bilinearly from the analysis grid (design: "再双线性放大") and
    // only that band is corrected: `gain · low + offset + (owner − low)`.
    // Pixels without a correction, without evidence or without any field cell
    // of their owner keep their exact value.
    let output_width = panorama.width();
    let output_height = panorama.height();
    let mut tile_of_owner =
        vec![None; tiles.iter().map(|tile| tile.owner_id).max().unwrap_or(0) as usize + 1];
    for (index, tile) in tiles.iter().enumerate() {
        tile_of_owner[tile.owner_id as usize].get_or_insert(index);
    }
    let corrects = |tile_index: usize| {
        let transform = &transforms[tile_index];
        transform.supported
            && (transform
                .gain
                .iter()
                .any(|value| (*value - 1.0).abs() > f32::EPSILON)
                || transform
                    .offset
                    .iter()
                    .any(|value| value.abs() > f32::EPSILON))
    };
    panorama
        .as_mut()
        .par_chunks_mut(output_width as usize * 3)
        .enumerate()
        .for_each(|(output_y, row)| {
            for (output_x, pixel) in row.chunks_exact_mut(3).enumerate() {
                let owner_id = owners[output_y * output_width as usize + output_x];
                let Some(tile_index) = tile_of_owner.get(owner_id as usize).copied().flatten()
                else {
                    continue;
                };
                if !corrects(tile_index) {
                    continue;
                }
                if let Some(evidence) = evidence
                    && (output_x as u32 >= evidence.width()
                        || output_y as u32 >= evidence.height()
                        || evidence.get_pixel(output_x as u32, output_y as u32)[0] == 0)
                {
                    continue;
                }
                let Some(low) = sample_tile_bilinear(
                    &tiles[tile_index],
                    world_origin.0 + output_x as f64,
                    world_origin.1 + output_y as f64,
                ) else {
                    continue;
                };
                let transform = &transforms[tile_index];
                for channel in 0..3 {
                    pixel[channel] = (pixel[channel]
                        + (transform.gain[channel] - 1.0) * low[channel]
                        + transform.offset[channel])
                        .clamp(0.0, 1.0);
                }
            }
        });

    let correct = |tile_index: usize, low: [f32; 3]| -> [f32; 3] {
        let transform = &transforms[tile_index];
        std::array::from_fn(|channel| {
            transform.gain[channel] * low[channel] + transform.offset[channel]
        })
    };
    // Tile pair and boundary window → corrected low sums of both tiles. Each
    // adjacent pair is read through its common-support fields, so a band next
    // to either tile's coverage edge compares the same content window.
    let mut pair_fields = std::collections::BTreeMap::<(usize, usize), Option<PairField>>::new();
    let mut boundary_windows =
        std::collections::BTreeMap::<(usize, usize, u32, u32), ([f64; 3], [f64; 3], u64)>::new();
    for y in 0..output_height {
        for x in 0..output_width {
            let index = y as usize * output_width as usize + x as usize;
            let Some(left_tile) = tile_of_owner.get(owners[index] as usize).copied().flatten()
            else {
                continue;
            };
            for (nx, ny) in [(x + 1, y), (x, y + 1)] {
                if nx >= output_width || ny >= output_height {
                    continue;
                }
                let right_index = ny as usize * output_width as usize + nx as usize;
                let Some(right_tile) = tile_of_owner
                    .get(owners[right_index] as usize)
                    .copied()
                    .flatten()
                else {
                    continue;
                };
                if right_tile == left_tile {
                    continue;
                }
                // The 16px bands on both sides of this boundary pixel, along
                // its normal. Both owners' corrected low fields are read at
                // the same positions, so the window means differ by the tone
                // step at the boundary and not by the artwork's own gradient
                // across a 32px strip (需求 9.8).
                let (first, second) = (left_tile.min(right_tile), left_tile.max(right_tile));
                let Some(field) = pair_fields
                    .entry((first, second))
                    .or_insert_with(|| PairField::new(&tiles[first], &tiles[second]))
                    .as_ref()
                else {
                    continue;
                };
                let window = (
                    first,
                    second,
                    x / TONE_BOUNDARY_WINDOW_PX,
                    y / TONE_BOUNDARY_WINDOW_PX,
                );
                for distance in 0..TONE_BOUNDARY_BAND_PX {
                    let near = if nx > x {
                        (x.checked_sub(distance), Some(y))
                    } else {
                        (Some(x), y.checked_sub(distance))
                    };
                    let far = if nx > x {
                        (
                            Some(nx + distance).filter(|&px| px < output_width),
                            Some(ny),
                        )
                    } else {
                        (
                            Some(nx),
                            Some(ny + distance).filter(|&py| py < output_height),
                        )
                    };
                    for position in [near, far] {
                        let (Some(px), Some(py)) = position else {
                            continue;
                        };
                        let Some((first_low, second_low)) = field.at(
                            world_origin.0 + f64::from(px),
                            world_origin.1 + f64::from(py),
                        ) else {
                            continue;
                        };
                        let (first_low, second_low) =
                            (correct(first, first_low), correct(second, second_low));
                        let entry = boundary_windows.entry(window).or_insert((
                            [0.0f64; 3],
                            [0.0f64; 3],
                            0u64,
                        ));
                        for channel in 0..3 {
                            entry.0[channel] += f64::from(first_low[channel]);
                            entry.1[channel] += f64::from(second_low[channel]);
                        }
                        entry.2 += 1;
                    }
                }
            }
        }
    }
    for ((first, second, window_x, window_y), (first_sum, second_sum, count)) in boundary_windows {
        let mean = |sum: [f64; 3]| sum.map(|channel| (channel / count as f64) as f32);
        let delta = delta_e00_rgb(mean(first_sum), mean(second_sum));
        report.boundary_delta_e.max = report.boundary_delta_e.max.max(delta);
        if delta <= TONE_BOUNDARY_DELTA_E_THRESHOLD {
            continue;
        }
        let world = super::report::WorldRect {
            left: world_origin.0 + f64::from(window_x * TONE_BOUNDARY_WINDOW_PX),
            top: world_origin.1 + f64::from(window_y * TONE_BOUNDARY_WINDOW_PX),
            width: f64::from(TONE_BOUNDARY_WINDOW_PX),
            height: f64::from(TONE_BOUNDARY_WINDOW_PX),
        };
        if report.boundary_delta_e.violations.len() < 4096 {
            report
                .boundary_delta_e
                .violations
                .push(ToneBoundaryViolationRecord {
                    left: tiles[first].station_index,
                    right: tiles[second].station_index,
                    delta_e00: delta,
                    world,
                });
        }
        degradation::record_run_degradation(
            degradation::TONE_BOUNDARY_DELTA_E_EXCEEDED,
            serde_json::json!({
                "left": tiles[first].station_index,
                "right": tiles[second].station_index,
                "delta_e00": delta,
                "x": world.left,
                "y": world.top,
                "window_px": TONE_BOUNDARY_WINDOW_PX,
                "samples": count,
            }),
        );
    }
    report.status = if solved_relations.is_empty() {
        ToneStatus::Identity
    } else if report.boundary_delta_e.violations.is_empty()
        && report.pairs_without_evidence.is_empty()
    {
        ToneStatus::Applied
    } else {
        ToneStatus::Degraded
    };
    for tile in &report.tiles {
        record_run_record(*tile);
    }
    record_run_report(report.clone());
    report
}

/// The tile's field at a world position, bilinear between the analysis-cell
/// centres the field was built at and renormalised over the neighbouring cells
/// that carry a field (validity ≠ 0). `None` where no neighbour has one.
fn sample_tile_bilinear(tile: &ToneTile, world_x: f64, world_y: f64) -> Option<[f32; 3]> {
    let stride = tile.world_stride;
    if !stride.is_finite() || stride <= 0.0 {
        return None;
    }
    let (width, height) = tile.low.dimensions();
    let u = (world_x - tile.world_origin.0 - (stride - 1.0) * 0.5) / stride;
    let v = (world_y - tile.world_origin.1 - (stride - 1.0) * 0.5) / stride;
    let (u0, v0) = (u.floor(), v.floor());
    let mut sum = [0.0f64; 3];
    let mut total = 0.0f64;
    for (dy, weight_y) in [(0.0, 1.0 - (v - v0)), (1.0, v - v0)] {
        for (dx, weight_x) in [(0.0, 1.0 - (u - u0)), (1.0, u - u0)] {
            let (x, y) = (u0 + dx, v0 + dy);
            if x < 0.0 || y < 0.0 || x >= f64::from(width) || y >= f64::from(height) {
                continue;
            }
            let (x, y) = (x as u32, y as u32);
            if x >= tile.validity.width()
                || y >= tile.validity.height()
                || tile.validity.get_pixel(x, y)[0] == 0
            {
                continue;
            }
            let weight = weight_x * weight_y;
            let value = tile.low.get_pixel(x, y).0;
            for channel in 0..3 {
                sum[channel] += weight * f64::from(value[channel]);
            }
            total += weight;
        }
    }
    (total > 1e-9).then(|| sum.map(|channel| (channel / total) as f32))
}

fn rgb_to_lab(rgb: [f32; 3]) -> [f64; 3] {
    let linear = rgb.map(|value| {
        let value = f64::from(value).clamp(0.0, 1.0);
        if value <= 0.04045 {
            value / 12.92
        } else {
            ((value + 0.055) / 1.055).powf(2.4)
        }
    });
    let x = (0.4124564 * linear[0] + 0.3575761 * linear[1] + 0.1804375 * linear[2]) / 0.95047;
    let y = 0.2126729 * linear[0] + 0.7151522 * linear[1] + 0.0721750 * linear[2];
    let z = (0.0193339 * linear[0] + 0.1191920 * linear[1] + 0.9503041 * linear[2]) / 1.08883;
    let f = |value: f64| {
        if value > 216.0 / 24389.0 {
            value.cbrt()
        } else {
            (24389.0 / 27.0 * value + 16.0) / 116.0
        }
    };
    let (fx, fy, fz) = (f(x), f(y), f(z));
    [116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)]
}

fn ciede2000(left: [f64; 3], right: [f64; 3]) -> f64 {
    let (l1, a1, b1) = (left[0], left[1], left[2]);
    let (l2, a2, b2) = (right[0], right[1], right[2]);
    let c1 = (a1 * a1 + b1 * b1).sqrt();
    let c2 = (a2 * a2 + b2 * b2).sqrt();
    let c_bar = (c1 + c2) * 0.5;
    let c_bar7 = c_bar.powi(7);
    let g = 0.5 * (1.0 - (c_bar7 / (c_bar7 + 25.0_f64.powi(7))).sqrt());
    let a1p = (1.0 + g) * a1;
    let a2p = (1.0 + g) * a2;
    let c1p = (a1p * a1p + b1 * b1).sqrt();
    let c2p = (a2p * a2p + b2 * b2).sqrt();
    let hp = |a: f64, b: f64| {
        if a == 0.0 && b == 0.0 {
            0.0
        } else {
            let mut angle = b.atan2(a).to_degrees();
            if angle < 0.0 {
                angle += 360.0;
            }
            angle
        }
    };
    let h1p = hp(a1p, b1);
    let h2p = hp(a2p, b2);
    let d_lp = l2 - l1;
    let d_cp = c2p - c1p;
    let dh = if c1p * c2p == 0.0 {
        0.0
    } else if (h2p - h1p).abs() <= 180.0 {
        h2p - h1p
    } else if h2p <= h1p {
        h2p - h1p + 360.0
    } else {
        h2p - h1p - 360.0
    };
    let d_hp = 2.0 * (c1p * c2p).sqrt() * (0.5 * (dh.to_radians())).sin();
    let l_bar = (l1 + l2) * 0.5;
    let c_bar_p = (c1p + c2p) * 0.5;
    let h_bar = if c1p * c2p == 0.0 {
        h1p + h2p
    } else if (h1p - h2p).abs() <= 180.0 {
        (h1p + h2p) * 0.5
    } else if h1p + h2p < 360.0 {
        (h1p + h2p + 360.0) * 0.5
    } else {
        (h1p + h2p - 360.0) * 0.5
    };
    let t = 1.0 - 0.17 * (h_bar - 30.0).to_radians().cos()
        + 0.24 * (2.0 * h_bar).to_radians().cos()
        + 0.32 * (3.0 * h_bar + 6.0).to_radians().cos()
        - 0.20 * (4.0 * h_bar - 63.0).to_radians().cos();
    let delta_theta = 30.0 * (-(h_bar - 275.0).powi(2) / 25.0_f64.powi(2)).exp();
    let rc = 2.0 * (c_bar_p.powi(7) / (c_bar_p.powi(7) + 25.0_f64.powi(7))).sqrt();
    let sl = 1.0 + 0.015 * (l_bar - 50.0).powi(2) / (20.0 + (l_bar - 50.0).powi(2)).sqrt();
    let sc = 1.0 + 0.045 * c_bar_p;
    let sh = 1.0 + 0.015 * c_bar_p * t;
    let rt = -(2.0 * delta_theta).to_radians().sin() * rc;
    ((d_lp / sl).powi(2)
        + (d_cp / sc).powi(2)
        + (d_hp / sh).powi(2)
        + rt * (d_cp / sc) * (d_hp / sh))
        .max(0.0)
        .sqrt()
}

fn bilinear_expand(low: &Rgb32FImage, width: u32, height: u32) -> Rgb32FImage {
    if low.width() == width && low.height() == height {
        return low.clone();
    }
    let x_scale = if width > 1 {
        (low.width().saturating_sub(1)) as f32 / (width - 1) as f32
    } else {
        0.0
    };
    let y_scale = if height > 1 {
        (low.height().saturating_sub(1)) as f32 / (height - 1) as f32
    } else {
        0.0
    };
    Rgb32FImage::from_fn(width, height, |x, y| {
        let fx = x as f32 * x_scale;
        let fy = y as f32 * y_scale;
        let x0 = fx.floor() as u32;
        let y0 = fy.floor() as u32;
        let x1 = (x0 + 1).min(low.width().saturating_sub(1));
        let y1 = (y0 + 1).min(low.height().saturating_sub(1));
        let tx = fx - x0 as f32;
        let ty = fy - y0 as f32;
        let top = low.get_pixel(x0, y0);
        let top_right = low.get_pixel(x1, y0);
        let bottom = low.get_pixel(x0, y1);
        let bottom_right = low.get_pixel(x1, y1);
        Rgb(std::array::from_fn(|channel| {
            let a = top[channel] * (1.0 - tx) + top_right[channel] * tx;
            let b = bottom[channel] * (1.0 - tx) + bottom_right[channel] * tx;
            a * (1.0 - ty) + b * ty
        }))
    })
}

fn median_f64(values: impl Iterator<Item = f64>) -> f64 {
    let mut values: Vec<_> = values.filter(|value| value.is_finite()).collect();
    if values.is_empty() {
        return 0.0;
    }
    values.sort_unstable_by(f64::total_cmp);
    let middle = values.len() / 2;
    if values.len().is_multiple_of(2) {
        (values[middle - 1] + values[middle]) * 0.5
    } else {
        values[middle]
    }
}

fn median(values: impl Iterator<Item = f32>) -> f32 {
    let mut values: Vec<_> = values.filter(|value| value.is_finite()).collect();
    if values.is_empty() {
        return 0.0;
    }
    values.sort_unstable_by(f32::total_cmp);
    let middle = values.len() / 2;
    if values.len().is_multiple_of(2) {
        (values[middle - 1] + values[middle]) * 0.5
    } else {
        values[middle]
    }
}

