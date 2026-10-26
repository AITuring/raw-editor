/// Measure one control point: build the two patches, match both ways and apply
/// every gate that does not depend on the other control points (需求 2.4 / 2.6).
#[allow(clippy::too_many_arguments)]
fn measure_native_control_point(
    base: &NativeRefineBase<'_>,
    sampler: &LayerSampler<'_>,
    x: u32,
    y: u32,
    spacing: f64,
    mode: NativeRefineMode,
    match_radius: i32,
    search_samples: i32,
) -> NativeControlPoint {
    let lx = x as f64 * sampler.scale;
    let ly = y as f64 * sampler.scale;
    // 需求 2.8 measures coverage against the anchor frame's valid pixel area;
    // the control point cells whose centre is covered are exactly that area,
    // discretised on this same grid.
    let centre_x = (sampler.left as f64 + lx) as u32;
    let centre_y = (sampler.top as f64 + ly) as u32;
    let mut measurement = NativeControlPoint {
        covered: base.covered(centre_x, centre_y),
        ..Default::default()
    };
    let patch_half = mode.patch_half();
    let side = (2 * patch_half + 1) as usize;
    let half = f64::from(patch_half);
    let mut base_patch = vec![0.0f64; side * side];
    let mut candidate_patch = vec![0.0f64; side * side];
    for py in 0..side {
        for px in 0..side {
            let dx = (px as f64 - half) * spacing;
            let dy = (py as f64 - half) * spacing;
            let gx = sampler.left as f64 + lx + dx;
            let gy = sampler.top as f64 + ly + dy;
            let Some(current) = base.masked(gx, gy) else {
                return measurement;
            };
            let Some(candidate) = sampler.sample(lx + dx, ly + dy) else {
                return measurement;
            };
            base_patch[py * side + px] = luma(current) * 255.0;
            candidate_patch[py * side + px] = luma(candidate) * 255.0;
        }
    }
    // A Capture_Station is a focus bracket: at this control point one of the two
    // sides is near its focal plane and the other is not, and raw-intensity
    // correlation cannot see two different point-spread functions as the same
    // content.  Bring them to a comparable scale first (需求 2.3).
    if mode.matches_focal_plane() {
        measurement.focal_plane = Some(intra_station::match_focal_plane(
            &mut base_patch,
            &mut candidate_patch,
            side,
        ));
    }
    let quantise = |values: &[f64]| {
        GrayImage::from_fn(side as u32, side as u32, |x, y| {
            Luma([values[y as usize * side + x as usize]
                .round()
                .clamp(0.0, 255.0) as u8])
        })
    };
    let a = quantise(&base_patch);
    let b = quantise(&candidate_patch);
    let gates = mode.patch_gates();
    let p = Point2::new(half, half);
    measurement.evaluated = true;
    let forward = probe_warped_patch(
        &a,
        &b,
        &Matrix3::identity(),
        p,
        match_radius,
        search_samples,
        gates,
    );
    measurement.best_correlation = forward.best_correlation;
    measurement.peak_margin = forward.peak_margin;
    let Some(found) = forward.matched else {
        return measurement;
    };
    // The mosaic path keeps its second, stricter correlation floor; the
    // intra-station path states its single floor in `gates` instead of gating
    // twice at two different values.
    if !mode.matches_focal_plane() && found.correlation < 0.90 {
        return measurement;
    }
    measurement.reverse_attempted = true;
    let Some(back) = probe_warped_patch(
        &b,
        &a,
        &Matrix3::identity(),
        found.target,
        match_radius,
        search_samples,
        gates,
    )
    .matched
    else {
        return measurement;
    };
    // Both residuals are patch-sample distances; one sample step is `spacing`
    // native pixels, so scaling by `spacing` puts every measurement below in
    // native pixels of the anchor frame.
    let round_trip_patch = (back.target - p).norm();
    let round_trip_px = round_trip_patch * spacing;
    measurement.round_trip_px = round_trip_px;
    // The symmetric reprojection error of the global model at this control point
    // is how far the two-way match says the frame has to move; after the
    // correction the remaining two-way disagreement is the round trip.  Both are
    // measurements of the completed bidirectional match, so they are recorded
    // before either gate looks at them.
    let global_error_px = (found.target - p).norm() * spacing;
    let d = (found.target - p) * (spacing / sampler.scale);
    measurement.measured = Some((
        [d.x, d.y],
        found.correlation,
        round_trip_px,
        global_error_px,
    ));
    measurement.round_trip_closed = match mode {
        // The legacy path compares the round trip in patch sample units,
        // exactly as it always did.
        NativeRefineMode::LegacyMosaic => {
            round_trip_patch <= intra_station::INTRA_STATION_BIDIRECTIONAL_TOLERANCE_PX
        }
        // 需求 2.4 measures in native pixels and allows 1.0; the stricter 0.40
        // of this path is kept and the measured value is reported.
        NativeRefineMode::IntraStation { .. } => {
            intra_station::bidirectional_consistent(round_trip_px)
        }
    };
    if !measurement.round_trip_closed {
        return measurement;
    }
    if let Some(anchor_long_side) = mode.anchor_long_side() {
        // 需求 2.6: an implausible displacement claim keeps the global model
        // here and counts as a rejected control point.
        if !intra_station::symmetric_error_accepted(global_error_px, anchor_long_side) {
            return measurement;
        }
    }
    measurement.accepted = measurement.measured;
    measurement
}

/// Measure every control point whose matching window lies inside `base`.
///
/// Refine against the selected, native output samples. Upsampling a coarse
/// displacement field alone leaves several pixels of error on 200MP input.
/// These tiny patches keep native refinement independent of canvas size.
///
/// The matching runs in parallel because each control point is an independent
/// measurement; the 需求 2.5 neighbourhood gate then walks the same grid
/// serially in row-major order, so the accepted set is exactly what a fully
/// serial walk accepts and is still a deterministic function of the
/// photographs.
fn collect_native_observations(
    base: &NativeRefineBase<'_>,
    sampler: &LayerSampler<'_>,
    width: u32,
    height: u32,
    spacing: f64,
    mode: NativeRefineMode,
    pass: &mut NativeRefinePass,
) {
    let (match_radius, search_samples) = mode.patch_geometry();
    debug_assert!(
        intra_station::patch_support_fits(match_radius, search_samples, mode.patch_half()),
        "the matching window and its search support must fit the patch buffer"
    );
    debug_assert!(
        !matches!(mode, NativeRefineMode::IntraStation { .. })
            || intra_station::native_search_reach_px(spacing)
                <= intra_station::INTRA_STATION_SEARCH_RADIUS,
        "需求 2.3 bounds the native search reach"
    );
    let columns = (20..width.saturating_sub(20))
        .step_by(NATIVE_REFINE_STEP as usize)
        .collect::<Vec<_>>();
    let rows = (20..height.saturating_sub(20))
        .step_by(NATIVE_REFINE_STEP as usize)
        .collect::<Vec<_>>();
    let measured = rows
        .par_iter()
        .map(|&y| {
            columns
                .iter()
                .map(|&x| {
                    measure_native_control_point(
                        base,
                        sampler,
                        x,
                        y,
                        spacing,
                        mode,
                        match_radius,
                        search_samples,
                    )
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    for (grid_row, row) in measured.iter().enumerate() {
        for (grid_column, measurement) in row.iter().enumerate() {
            if measurement.covered {
                pass.valid_cells += 1;
            }
            if !measurement.evaluated {
                continue;
            }
            pass.evaluated += 1;
            pass.similarity.absorb(measurement);
            pass.max_round_trip = pass.max_round_trip.max(measurement.round_trip_px);
            let Some((d, correlation, local_error_px, global_error_px)) = measurement.accepted
            else {
                continue;
            };
            if mode.anchor_long_side().is_some() {
                // 需求 2.5: agree with the already accepted 8-neighbourhood, in
                // native pixels.
                let native = [d[0] * sampler.scale, d[1] * sampler.scale];
                if !intra_station::neighbour_median_consistent(
                    native,
                    pass.accepted_grid.neighbour_median(grid_column, grid_row),
                ) {
                    continue;
                }
                pass.accepted_grid.accept(grid_column, grid_row, native);
            }
            pass.global_errors.push(global_error_px);
            pass.local_errors.push(local_error_px);
            pass.observations.push((
                Point2::new(columns[grid_column] as f64, rows[grid_row] as f64),
                d,
                correlation,
            ));
        }
    }
}

/// Fold one spacing pass into the outcome and, when it carries enough accepted
/// control points, into the sampler's displacement field.
#[allow(clippy::too_many_arguments)]
fn apply_native_pass(
    mut pass: NativeRefinePass,
    sampler: &mut LayerSampler<'_>,
    outcome: &mut NativeRefineOutcome,
    accepted_cells: &mut intra_station::ControlPointGrid,
    cell_area_px: f64,
    spacing: f64,
    is_entry_pass: bool,
    mode: NativeRefineMode,
) {
    accepted_cells.absorb(&pass.accepted_grid);
    // The legacy comparison path never populated the grid and never reported
    // these counts, so it keeps counting one pass at a time.
    let accepted = if mode == NativeRefineMode::LegacyMosaic {
        pass.observations.len()
    } else {
        accepted_cells.accepted_count()
    };
    outcome.evaluated_control_points = pass.evaluated;
    outcome.accepted_control_points = accepted;
    outcome.rejected_control_point_ratio = if pass.evaluated == 0 {
        0.0
    } else {
        1.0 - outcome.accepted_control_points as f64 / pass.evaluated as f64
    };
    outcome.max_bidirectional_round_trip_px = pass.max_round_trip;
    // 需求 2.7 compares the refined frame against the *global* model, so the
    // baseline has to be measured while the global model is still what the
    // sampler carries.  That is the entry pass — but a pass whose accepted
    // control points are too few to form a field leaves the sampler untouched,
    // so the *next* pass is still measuring the global model and is the first
    // one that can supply a baseline.  Claiming a baseline from a pass that
    // supplied none is how the comparison ended up against 0.00px.
    let no_baseline_yet =
        outcome.global_baseline_samples < intra_station::INTRA_STATION_MIN_BASELINE_SAMPLES;
    if (is_entry_pass || no_baseline_yet) && !pass.global_errors.is_empty() {
        outcome.global_baseline_samples = pass.global_errors.len();
        outcome.global_median_symmetric_error_px =
            intra_station::median_of(&mut pass.global_errors);
    }
    outcome.local_baseline_samples = pass.local_errors.len();
    outcome.local_median_symmetric_error_px = intra_station::median_of(&mut pass.local_errors);
    outcome.inlier_area_coverage = intra_station::inlier_area_coverage(
        accepted,
        cell_area_px,
        pass.valid_cells as f64 * cell_area_px,
    );
    #[cfg(test)]
    {
        outcome.last_pass_accepted_control_points = pass.accepted_grid.accepted_count();
        outcome.valid_cells = pass.valid_cells;
        outcome.cell_area_px = cell_area_px;
    }
    let (focal_before, focal_after) = pass.similarity.median_focal_ratios();
    outcome.focal_plane_ratio_before = focal_before;
    outcome.focal_plane_ratio_after = focal_after;
    println!(
        "    - Patch similarity ({spacing:.0}px samples): {}",
        pass.similarity.summary()
    );
    if pass.observations.len() < intra_station::INTRA_STATION_MIN_BASELINE_SAMPLES {
        return;
    }
    let observations = &pass.observations;
    let mut field = sampler.residual.clone();
    for gy in 0..field.height {
        for gx in 0..field.width {
            let p = Point2::new(gx as f64 * field.step, gy as f64 * field.step);
            let mut sum = [0.0; 2];
            let mut total = 0.0f64;
            for (q, d, correlation) in observations {
                let weight = (-(p - q).norm_squared()
                    / (2.0 * NATIVE_FIELD_RADIUS * NATIVE_FIELD_RADIUS))
                    .exp()
                    * correlation.powi(8);
                sum[0] += weight * d[0];
                sum[1] += weight * d[1];
                total += weight;
            }
            for (c, value) in sum.into_iter().enumerate() {
                field.values[gy * field.width + gx][c] += value / total.max(0.1);
            }
        }
    }
    sampler.residual = field;
    let correction = median(
        &mut observations
            .iter()
            .map(|(_, d, _)| d[0].hypot(d[1]) * sampler.scale)
            .collect::<Vec<_>>(),
    );
    println!(
        "    - Native refinement ({spacing:.0}px samples): {} patches, median correction {correction:.2}px",
        observations.len()
    );
}

/// 需求 2.7 / 2.8: the per-frame verdict of a finished native refinement.
fn finalize_native_refinement(
    outcome: &mut NativeRefineOutcome,
    sampler: &mut LayerSampler<'_>,
    entry_field: Field<2>,
    mode: NativeRefineMode,
) {
    if mode == NativeRefineMode::LegacyMosaic {
        return;
    }
    // 需求 2.7: keep the local field only when it lowered the frame's median
    // symmetric reprojection error.  A refinement that merely moved the error
    // around is a worse description of the same photographs than the global
    // model, so the whole frame goes back to the state it entered with.  When
    // one of the two medians has too few samples to exist the comparison is
    // unevaluable, and the requirement's condition — "not lower than" — is not
    // satisfied by an absent value, so nothing is reverted on its behalf.
    outcome.local_field_verdict = intra_station::local_field_verdict(
        outcome.local_median_symmetric_error_px,
        outcome.local_baseline_samples,
        outcome.global_median_symmetric_error_px,
        outcome.global_baseline_samples,
    );
    match outcome.local_field_verdict {
        intra_station::LocalFieldVerdict::Kept => {}
        intra_station::LocalFieldVerdict::Reverted => {
            sampler.residual = entry_field;
            outcome.reverted_to_global = true;
            println!(
                "    - Native refinement reverted: median symmetric error {:.2}px over {} points is not better than the global model's {:.2}px over {} points",
                outcome.local_median_symmetric_error_px,
                outcome.local_baseline_samples,
                outcome.global_median_symmetric_error_px,
                outcome.global_baseline_samples
            );
            degradation::record_run_degradation(
                degradation::INTRA_STATION_LOCAL_FALLBACK,
                serde_json::json!({
                    "stage": "intra_station_native_refinement",
                    "local_median_symmetric_error_px": outcome.local_median_symmetric_error_px,
                    "local_baseline_samples": outcome.local_baseline_samples,
                    "global_median_symmetric_error_px": outcome.global_median_symmetric_error_px,
                    "global_baseline_samples": outcome.global_baseline_samples,
                    "accepted_control_points": outcome.accepted_control_points,
                }),
            );
        }
        intra_station::LocalFieldVerdict::Indeterminate => {
            println!(
                "    - Native refinement indeterminate: {} local and {} global control points cannot form the 需求 2.7 comparison; the measured field is kept",
                outcome.local_baseline_samples, outcome.global_baseline_samples
            );
            degradation::record_run_degradation(
                degradation::INTRA_STATION_LOCAL_BASELINE_INDETERMINATE,
                serde_json::json!({
                    "stage": "intra_station_native_refinement",
                    "local_median_symmetric_error_px": outcome.local_median_symmetric_error_px,
                    "local_baseline_samples": outcome.local_baseline_samples,
                    "global_median_symmetric_error_px": outcome.global_median_symmetric_error_px,
                    "global_baseline_samples": outcome.global_baseline_samples,
                    "minimum_baseline_samples":
                        intra_station::INTRA_STATION_MIN_BASELINE_SAMPLES,
                    "accepted_control_points": outcome.accepted_control_points,
                }),
            );
        }
    }
    // 需求 2.8: a frame whose accepted control point cells cover less than 20%
    // of the anchor frame's valid pixel area is a registration failure.
    if outcome.inlier_area_coverage < intra_station::INTRA_STATION_MIN_INLIER_AREA_COVERAGE {
        println!(
            "    - Intra-station registration failed: inlier area coverage {:.3} below {:.2}",
            outcome.inlier_area_coverage,
            intra_station::INTRA_STATION_MIN_INLIER_AREA_COVERAGE
        );
        degradation::record_run_degradation(
            degradation::INTRA_STATION_REGISTRATION_FAILED,
            serde_json::json!({
                "stage": "intra_station_native_refinement",
                "inlier_area_coverage": outcome.inlier_area_coverage,
                "minimum_inlier_area_coverage":
                    intra_station::INTRA_STATION_MIN_INLIER_AREA_COVERAGE,
                "accepted_control_points": outcome.accepted_control_points,
                "evaluated_control_points": outcome.evaluated_control_points,
            }),
        );
    }
}

