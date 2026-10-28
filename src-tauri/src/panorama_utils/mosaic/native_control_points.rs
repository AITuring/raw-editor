//! Native control-point measurement of the Intra_Station_Registrar.
//!
//! Each control point is matched both ways and gated on its own, so matching
//! runs in parallel while the neighbourhood gate stays a serial row-major walk;
//! correlation buckets record the similarity/round-trip tradeoff.

use super::super::registration::probe_warped_patch;
use super::super::stack_pipeline::intra_station;
use image::GrayImage;
use image::Luma;
use nalgebra::Matrix3;
use nalgebra::Point2;
use rayon::prelude::*;

use super::{
    LayerSampler, NATIVE_REFINE_STEP, NativeRefineBase, NativeRefineMode, NativeRefinePass, luma,
};

/// Width of one correlation bucket, and the value the lowest bucket starts at.
pub(super) const SIMILARITY_BUCKET_WIDTH: f64 = 0.05;

pub(super) const SIMILARITY_BUCKET_FLOOR: f64 = 0.50;

pub(super) const SIMILARITY_BUCKETS: usize = 10;

/// Correlation the round-trip distribution is reported from, so the log states
/// the tradeoff at the similarity floor that is actually in force.
pub(super) const SIMILARITY_REPORTED_FLOOR: f64 = intra_station::INTRA_STATION_MIN_CORRELATION;

/// Which correlation bucket a score belongs to, or `None` below the floor.
pub(super) fn similarity_bucket(correlation: f64) -> Option<usize> {
    if !correlation.is_finite() || correlation < SIMILARITY_BUCKET_FLOOR {
        return None;
    }
    Some(
        (((correlation - SIMILARITY_BUCKET_FLOOR) / SIMILARITY_BUCKET_WIDTH) as usize)
            .min(SIMILARITY_BUCKETS - 1),
    )
}

/// What the similarity gates of one pass actually saw.
///
/// This exists so the correlation threshold can be stated as a measured
/// accept-rate/precision tradeoff rather than as a number someone liked.  The
/// precision proxy is the share of each correlation bucket that then closed the
/// 需求 2.4 bidirectional round trip within 0.40 native pixels: a bucket whose
/// forward peak is real closes the round trip, a bucket matching a repeated
/// texture does not.
#[derive(Debug, Clone, Default)]
pub(super) struct SimilarityStats {
    /// Forward peak correlation of every control point whose patch pair carried
    /// comparable signal at all.
    pub(super) correlations: Vec<f64>,
    /// Per bucket, how many control points reached the reverse match.
    pub(super) bucket_total: [usize; SIMILARITY_BUCKETS],
    /// Per bucket, how many of those closed the round trip.
    pub(super) bucket_round_trip: [usize; SIMILARITY_BUCKETS],
    /// Round trip in native pixels of every control point that reached the
    /// reverse match at or above [`SIMILARITY_REPORTED_FLOOR`], so the 需求 2.4
    /// tolerance can be read off a distribution instead of guessed.
    pub(super) accepted_round_trips: Vec<f64>,
    /// Second-peak margin of those same control points, so the margin gate can
    /// be checked against data too.
    pub(super) accepted_peak_margins: Vec<f64>,
    /// `softer / sharper` high-frequency energy ratio of every measured pair,
    /// before and after the matched low-pass.
    pub(super) focal_ratio_before: Vec<f64>,
    pub(super) focal_ratio_matched: Vec<f64>,
    pub(super) focal_ratio_after: Vec<f64>,
    /// Sigma, in patch samples, the focal-plane match decided the two sides
    /// differ by, and how often the descent ran out of budget.
    pub(super) focal_passes: Vec<f64>,
    pub(super) focal_capped: usize,
}

impl SimilarityStats {
    pub(super) fn absorb(&mut self, point: &NativeControlPoint) {
        if let Some(correlation) = point.best_correlation {
            self.correlations.push(correlation);
            if let Some(bucket) = similarity_bucket(correlation)
                && point.reverse_attempted
            {
                self.bucket_total[bucket] += 1;
                if point.round_trip_closed {
                    self.bucket_round_trip[bucket] += 1;
                }
                if correlation >= SIMILARITY_REPORTED_FLOOR {
                    self.accepted_round_trips.push(point.round_trip_px);
                    if let Some(margin) = point.peak_margin {
                        self.accepted_peak_margins.push(margin);
                    }
                }
            }
        }
        if let Some(focal) = point.focal_plane {
            self.focal_ratio_before.push(focal.sharpness_ratio_before);
            self.focal_ratio_matched.push(focal.sharpness_ratio_matched);
            self.focal_ratio_after.push(focal.sharpness_ratio_after);
            self.focal_passes.push(focal.matched_variance.sqrt());
            self.focal_capped += usize::from(focal.capped);
        }
    }

    /// One log line: the correlation distribution, the measured round-trip pass
    /// rate per bucket and what the focal-plane match did.
    pub(super) fn summary(&mut self) -> String {
        if self.correlations.is_empty() {
            return "no measurable patch pair".to_string();
        }
        let percentile = |values: &mut Vec<f64>, fraction: f64| -> f64 {
            values.sort_by(|left, right| left.total_cmp(right));
            let index = ((values.len() - 1) as f64 * fraction).round() as usize;
            values[index]
        };
        let p10 = percentile(&mut self.correlations, 0.10);
        let p50 = percentile(&mut self.correlations, 0.50);
        let p90 = percentile(&mut self.correlations, 0.90);
        let buckets = (0..SIMILARITY_BUCKETS)
            .filter(|&bucket| self.bucket_total[bucket] > 0)
            .map(|bucket| {
                format!(
                    "{:.2}:{}/{}",
                    SIMILARITY_BUCKET_FLOOR + bucket as f64 * SIMILARITY_BUCKET_WIDTH,
                    self.bucket_round_trip[bucket],
                    self.bucket_total[bucket]
                )
            })
            .collect::<Vec<_>>()
            .join(" ");
        let focal = if self.focal_ratio_before.is_empty() {
            String::new()
        } else {
            format!(
                ", focal-plane ratio p50 {:.3} -> {:.3} -> {:.3} at sigma p50/p90 {:.2}/{:.2} samples, {} capped",
                percentile(&mut self.focal_ratio_before, 0.50),
                percentile(&mut self.focal_ratio_matched, 0.50),
                percentile(&mut self.focal_ratio_after, 0.50),
                percentile(&mut self.focal_passes, 0.50),
                percentile(&mut self.focal_passes, 0.90),
                self.focal_capped
            )
        };
        let round_trip = if self.accepted_round_trips.is_empty() {
            String::new()
        } else {
            let total = self.accepted_round_trips.len();
            let round_trips = &mut self.accepted_round_trips;
            let p50 = percentile(round_trips, 0.50);
            let p90 = percentile(round_trips, 0.90);
            let within = |limit: f64| round_trips.iter().filter(|&&value| value <= limit).count();
            let counted = format!(
                ", round trip at correlation >= {SIMILARITY_REPORTED_FLOOR:.2} p50/p90 {p50:.2}/{p90:.2}px, within 0.40/0.70/1.00px {}/{}/{} of {total}",
                within(0.40),
                within(0.70),
                within(1.00)
            );
            let margin = percentile(&mut self.accepted_peak_margins, 0.10);
            format!("{counted}, second-peak margin p10 {margin:.4}")
        };
        format!(
            "correlation p10/p50/p90 {p10:.3}/{p50:.3}/{p90:.3} over {} pairs{focal}{round_trip}, round trip closure by bucket [{buckets}]",
            self.correlations.len()
        )
    }

    pub(super) fn median_focal_ratios(&mut self) -> (f64, f64) {
        (
            if self.focal_ratio_before.is_empty() {
                1.0
            } else {
                intra_station::median_of(&mut self.focal_ratio_before)
            },
            if self.focal_ratio_after.is_empty() {
                1.0
            } else {
                intra_station::median_of(&mut self.focal_ratio_after)
            },
        )
    }
}

/// What one control point measured, before the order dependent 需求 2.5 gate.
///
/// Every field here is a function of that control point alone, which is what
/// lets the expensive matching run in parallel while the neighbourhood gate
/// stays a serial row-major walk.
#[derive(Debug, Clone, Copy, Default)]
pub(super) struct NativeControlPoint {
    /// The cell centre lies on covered base, so its area counts towards the
    /// anchor frame's valid pixel area (需求 2.8).
    pub(super) covered: bool,
    /// The patch pair was complete, so a match was attempted.
    pub(super) evaluated: bool,
    /// Bidirectional round trip in native pixels, `0.0` when never measured.
    pub(super) round_trip_px: f64,
    /// Forward peak correlation, `None` when the patch pair carried no
    /// comparable signal at all.
    pub(super) best_correlation: Option<f64>,
    /// `true` when the forward match cleared the similarity gates, so a reverse
    /// match was run and the round trip is a real measurement.
    pub(super) reverse_attempted: bool,
    /// `true` when that round trip closed inside the 需求 2.4 tolerance.
    pub(super) round_trip_closed: bool,
    /// Gap between the correlation peak and the best hypothesis at least 2.5
    /// samples away, `None` when no hypothesis was scored.
    pub(super) peak_margin: Option<f64>,
    /// What the focal-plane matched low-pass did to this pair.
    pub(super) focal_plane: Option<intra_station::FocalPlaneMatch>,
    /// Displacement in analysis units, its correlation, the local (round trip)
    /// and global symmetric reprojection errors in native pixels.  `Some` as soon
    /// as both directions matched, i.e. *before* the 需求 2.4 and 需求 2.6 gates.
    ///
    /// Published separately from [`Self::accepted`] so the gates can be stated as
    /// a filter over a measurement rather than as the measurement itself.
    pub(super) measured: Option<([f64; 2], f64, f64, f64)>,
    /// [`Self::measured`] once the 需求 2.4 and 需求 2.6 gates have passed.
    pub(super) accepted: Option<([f64; 2], f64, f64, f64)>,
}

/// Measure one control point: build the two patches, match both ways and apply
/// every gate that does not depend on the other control points (需求 2.4 / 2.6).
#[allow(clippy::too_many_arguments)]
pub(super) fn measure_native_control_point(
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
/// Keep one observation per control cell, but let a cell whose centre misses
/// a small high-contrast object sample that object within its own cell. Every
/// relocated patch still passes the ordinary correlation/round-trip gates.
pub(super) fn contrasted_native_control_location(
    base: &NativeRefineBase<'_>,
    sampler: &LayerSampler<'_>,
    x: u32,
    y: u32,
    width: u32,
    height: u32,
) -> (u32, u32) {
    let gx = sampler.left as f64 + x as f64 * sampler.scale;
    let gy = sampler.top as f64 + y as f64 * sampler.scale;
    if !base.covered(gx as u32, gy as u32)
        || sampler
            .sample(x as f64 * sampler.scale, y as f64 * sampler.scale)
            .is_none()
    {
        return (x, y);
    }
    let contrast = |cx: u32, cy: u32| {
        let cx = sampler.left as f64 + cx as f64 * sampler.scale;
        let cy = sampler.top as f64 + cy as f64 * sampler.scale;
        let mut maximum = 0.0f64;
        let (mut xx, mut xy, mut yy) = (0.0f64, 0.0f64, 0.0f64);
        for dy in [-16.0, -8.0, 0.0, 8.0, 16.0] {
            for dx in [-16.0, -8.0, 0.0, 8.0, 16.0] {
                let (px, py) = (cx + dx, cy + dy);
                if let (Some(l), Some(r), Some(t), Some(b)) = (
                    base.masked(px - 1.0, py),
                    base.masked(px + 1.0, py),
                    base.masked(px, py - 1.0),
                    base.masked(px, py + 1.0),
                ) {
                    let gx = f64::from(luma(r) - luma(l)) * 0.5;
                    let gy = f64::from(luma(b) - luma(t)) * 0.5;
                    maximum = maximum.max(gx.hypot(gy));
                    xx += gx * gx;
                    xy += gx * gy;
                    yy += gy * gy;
                }
            }
        }
        let corner = ((xx + yy - ((xx - yy).powi(2) + 4.0 * xy * xy).sqrt()) * 0.5).max(0.0);
        (maximum, corner)
    };
    let mut best = (contrast(x, y).1, x, y);
    let quarter = NATIVE_REFINE_STEP as i32 / 4;
    for dy in [-quarter, 0, quarter] {
        for dx in [-quarter, 0, quarter] {
            let (px, py) = (x as i32 + dx, y as i32 + dy);
            if px < 20
                || py < 20
                || px >= width.saturating_sub(20) as i32
                || py >= height.saturating_sub(20) as i32
            {
                continue;
            }
            let value = contrast(px as u32, py as u32);
            if value.0 >= 0.15 && value.1 > best.0 * 1.25 + 1e-8 {
                best = (value.1, px as u32, py as u32);
            }
        }
    }
    (best.1, best.2)
}

pub(super) fn collect_native_observations(
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
                    let (px, py) = if mode.anchor_long_side().is_some() && spacing <= 1.0 {
                        contrasted_native_control_location(base, sampler, x, y, width, height)
                    } else {
                        (x, y)
                    };
                    let mut point = measure_native_control_point(
                        base,
                        sampler,
                        px,
                        py,
                        spacing,
                        mode,
                        match_radius,
                        search_samples,
                    );
                    let mut position = (px, py);
                    if position != (x, y) && point.accepted.is_none() {
                        point = measure_native_control_point(
                            base,
                            sampler,
                            x,
                            y,
                            spacing,
                            mode,
                            match_radius,
                            search_samples,
                        );
                        position = (x, y);
                    }
                    // Coverage remains the original cell-centre quadrature;
                    // moving the matching patch cannot enlarge spatial support.
                    point.covered = base.covered(
                        (sampler.left as f64 + x as f64 * sampler.scale) as u32,
                        (sampler.top as f64 + y as f64 * sampler.scale) as u32,
                    );
                    (position, point)
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    for (grid_row, row) in measured.iter().enumerate() {
        for (grid_column, (position, measurement)) in row.iter().enumerate() {
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
                Point2::new(position.0 as f64, position.1 as f64),
                d,
                correlation,
            ));
        }
    }
}
