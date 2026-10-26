/// Sharpness_Score of a gray analysis image, sampled with `spacing` pixels
/// between probes so a measurement can be taken at a bounded analysis
/// resolution (需求 2.1 / 2.2).  `None` where the probe window would leave the
/// image, which is what "valid pixels" means for a full frame.
pub(crate) fn gray_analysis_sharpness(
    image: &GrayImage,
    x: f64,
    y: f64,
    spacing: f64,
) -> Option<f64> {
    let half = 6.0 * spacing + 1.0;
    if !(x - half >= 0.0
        && y - half >= 0.0
        && x + half < f64::from(image.width())
        && y + half < f64::from(image.height()))
    {
        return None;
    }
    Some(acutance(
        |sample_x, sample_y| {
            intra_station::gray_sample(
                image,
                x + (sample_x - x) * spacing,
                y + (sample_y - y) * spacing,
            )
        },
        x,
        y,
    ))
}

/// Which caller the native patch refinement is serving.
///
/// The intra-station registrar composites at native resolution by definition,
/// so the analysis-scale early exit has no meaning there (需求 2.2).  The legacy
/// mosaic comparison path keeps it, together with its original search radius
/// and its single bidirectional gate, so its output stays byte identical.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum NativeRefineMode {
    /// Legacy comparison path: `scale <= 2.0` returns without measuring.
    LegacyMosaic,
    /// Intra-station path: every control point passes the 需求 2.4/2.5/2.6 gates
    /// and the frame keeps its local field only when it improves (需求 2.7).
    IntraStation {
        /// Native long side of the anchor frame, for the 需求 2.6 error gate.
        anchor_long_side: u32,
    },
}

impl NativeRefineMode {
    /// `(matching window radius, translation search extent)` in patch samples.
    ///
    /// The legacy pair is unchanged.  The intra-station pair widens the window to
    /// 33 native pixels at the finest pass (需求 2.3) and keeps the search support
    /// inside the patch buffer, which is the only way `refine_warped_patch`
    /// returns a match at all.
    fn patch_geometry(self) -> (i32, i32) {
        match self {
            Self::LegacyMosaic => (10, 7),
            Self::IntraStation { .. } => (
                intra_station::INTRA_STATION_MATCH_RADIUS,
                intra_station::INTRA_STATION_SEARCH_SAMPLES,
            ),
        }
    }

    fn anchor_long_side(self) -> Option<u32> {
        match self {
            Self::LegacyMosaic => None,
            Self::IntraStation { anchor_long_side } => Some(anchor_long_side),
        }
    }

    /// Half side of the patch buffers this mode builds.
    fn patch_half(self) -> i32 {
        match self {
            Self::LegacyMosaic => intra_station::LEGACY_MOSAIC_PATCH_HALF,
            Self::IntraStation { .. } => intra_station::INTRA_STATION_PATCH_HALF,
        }
    }

    /// Similarity gates of one control point.
    fn patch_gates(self) -> PatchMatchGates {
        match self {
            Self::LegacyMosaic => PatchMatchGates::LEGACY,
            Self::IntraStation { .. } => intra_station::INTRA_STATION_PATCH_GATES,
        }
    }

    /// `true` when the two sides of a patch pair have to be brought to a
    /// comparable focal plane before they are correlated.
    ///
    /// Only the intra-station path needs it, because only there are the two
    /// sides deliberately focused on different planes.  The mosaic path matches
    /// two frames of the same plane and its output has to stay byte identical.
    fn matches_focal_plane(self) -> bool {
        matches!(self, Self::IntraStation { .. })
    }
}

/// What one native refinement pass measured (需求 2.9).  Every field is an
/// observation; the caller turns it into the report record.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct NativeRefineOutcome {
    /// Control points that passed every gate in the final pass.
    pub(crate) accepted_control_points: usize,
    /// Control points whose measurement was attempted in the final pass.
    pub(crate) evaluated_control_points: usize,
    /// `1 - accepted / evaluated` (需求 2.6).
    pub(crate) rejected_control_point_ratio: f64,
    /// Median symmetric reprojection error of the global model, native pixels.
    pub(crate) global_median_symmetric_error_px: f64,
    /// Control points the global baseline median was formed from.  `0` means
    /// 需求 2.7 has no baseline to compare against (需求 2.7 / 2.9).
    pub(crate) global_baseline_samples: usize,
    /// Median symmetric reprojection error after refinement, native pixels.
    pub(crate) local_median_symmetric_error_px: f64,
    /// Control points the refined median was formed from.
    pub(crate) local_baseline_samples: usize,
    /// How 需求 2.7 resolved (需求 2.9).
    pub(crate) local_field_verdict: intra_station::LocalFieldVerdict,
    /// Median `softer / sharper` high-frequency energy ratio of the control
    /// point patch pairs before and after the focal-plane matched low-pass.
    pub(crate) focal_plane_ratio_before: f64,
    pub(crate) focal_plane_ratio_after: f64,
    /// Largest observed bidirectional round trip, native pixels (需求 2.4).
    pub(crate) max_bidirectional_round_trip_px: f64,
    /// 需求 2.8 coverage of the accepted control point cells.
    pub(crate) inlier_area_coverage: f64,
    /// Measured distance between neighbouring control points, in native pixels
    /// of the anchor frame (需求 2.3).  The grid step is fixed in *analysis*
    /// units, so the native spacing is whatever the layer's analysis scale makes
    /// it, and only a measurement can say what that was.
    pub(crate) control_point_spacing_px: f64,
    /// `true` when the whole local field was discarded (需求 2.7).
    pub(crate) reverted_to_global: bool,
    /// `true` when the mode declined to measure at all.
    pub(crate) skipped: bool,
    /// Accepted control point cells of the *final* spacing pass alone.
    ///
    /// Test-only, and the whole point of it is to be different from
    /// `accepted_control_points`: 需求 2.8 counts the cells that contain an
    /// accepted local match over the *whole* refinement, so the coverage is the
    /// union over the spacing passes ([`intra_station::ControlPointGrid::absorb`])
    /// while this is what the last pass on its own saw.  Property 8 asserts which
    /// of the two the reported coverage is built from, and it can only do that if
    /// it can see both.
    #[cfg(test)]
    pub(crate) last_pass_accepted_control_points: usize,
    /// Control point cells whose centre lay on covered base in the final pass,
    /// i.e. the denominator of 需求 2.8 in cells.  Test-only, for the same
    /// reason.
    #[cfg(test)]
    pub(crate) valid_cells: usize,
    /// Area of one control point cell in native pixels of the anchor frame.
    /// Test-only, for the same reason.
    #[cfg(test)]
    pub(crate) cell_area_px: f64,
}

impl Default for NativeRefineOutcome {
    fn default() -> Self {
        Self {
            accepted_control_points: 0,
            evaluated_control_points: 0,
            rejected_control_point_ratio: 0.0,
            global_median_symmetric_error_px: 0.0,
            global_baseline_samples: 0,
            local_median_symmetric_error_px: 0.0,
            local_baseline_samples: 0,
            local_field_verdict: intra_station::LocalFieldVerdict::Indeterminate,
            focal_plane_ratio_before: 1.0,
            focal_plane_ratio_after: 1.0,
            max_bidirectional_round_trip_px: 0.0,
            inlier_area_coverage: 0.0,
            control_point_spacing_px: 0.0,
            reverted_to_global: false,
            skipped: true,
            #[cfg(test)]
            last_pass_accepted_control_points: 0,
            #[cfg(test)]
            valid_cells: 0,
            #[cfg(test)]
            cell_area_px: 0.0,
        }
    }
}

impl NativeRefineOutcome {
    /// 需求 2.7 / 2.8: the frame verdict implied by this measurement.
    pub(crate) fn frame_status(&self) -> IntraStationFrameStatus {
        intra_station::frame_status(
            self.local_field_verdict.field_kept(),
            self.inlier_area_coverage,
        )
    }
}

/// One `intra_station[*].frames[*]` entry (需求 2.9).  Shared by the in-memory
/// and the tiled compositor so the report cannot describe one of them only.
fn intra_station_frame_record(
    path: &str,
    into_anchor: Matrix3<f64>,
    outcome: &NativeRefineOutcome,
) -> IntraStationFrameRecord {
    IntraStationFrameRecord {
        path: path.to_string(),
        transform: [
            into_anchor[(0, 0)],
            into_anchor[(0, 1)],
            into_anchor[(0, 2)],
            into_anchor[(1, 0)],
            into_anchor[(1, 1)],
            into_anchor[(1, 2)],
            into_anchor[(2, 0)],
            into_anchor[(2, 1)],
            into_anchor[(2, 2)],
        ],
        inliers: outcome.accepted_control_points,
        inlier_area_coverage: outcome.inlier_area_coverage,
        control_point_spacing_px: outcome.control_point_spacing_px,
        rejected_control_point_ratio: outcome.rejected_control_point_ratio,
        median_symmetric_error_px: if outcome.reverted_to_global {
            outcome.global_median_symmetric_error_px
        } else {
            outcome.local_median_symmetric_error_px
        },
        global_median_symmetric_error_px: outcome.global_median_symmetric_error_px,
        global_baseline_samples: outcome.global_baseline_samples,
        local_baseline_samples: outcome.local_baseline_samples,
        local_field_verdict: outcome.local_field_verdict.record(),
        focal_plane_ratio_before: outcome.focal_plane_ratio_before,
        focal_plane_ratio_after: outcome.focal_plane_ratio_after,
        status: outcome.frame_status(),
    }
}

/// Median Sharpness_Score over the valid pixels of one Source_RAW, measured the
/// same way the anchor selection of 需求 2.1 measures it.
///
/// This is what 需求 12.1 ranks the members of a degraded Capture_Station by.
/// The measurement runs on the already prepared analysis image and on the fixed
/// probe grid of [`intra_station::median_valid_sharpness`], so it costs a few
/// thousand samples per frame and is independent of the canvas the station
/// happens to be rendered on.
///
/// Shared with the default station path in [`super::stitching`], so a degraded
/// station ranks its members by the same number whichever compositor rendered
/// it.
pub(crate) fn station_member_median_sharpness(info: &ImageInfo) -> f64 {
    let analysis = &info.alignment_image;
    let (width, height) = (analysis.width(), analysis.height());
    let spacing = intra_station::analysis_probe_spacing(width.max(height));
    intra_station::median_valid_sharpness(width, height, |x, y| {
        gray_analysis_sharpness(analysis, x, y, spacing)
    })
}

/// Masked base pixels of the already composited canvas, addressed in global
/// canvas coordinates.
///
/// The non-streaming path wraps the whole canvas.  The streaming path wraps one
/// [`STREAMING_TILE_SIZE`] tile at a time — a canvas of hundreds of megapixels
/// never exists in memory there — so a control point whose matching window
/// leaves the loaded tile has no base to measure against and is skipped, exactly
/// as a control point over uncovered canvas is.  The residual field is a smooth
/// Gaussian interpolation of the accepted observations, so the thin band along
/// each tile border is carried by its neighbours.
struct NativeRefineBase<'a> {
    image: &'a Rgb32FImage,
    mask: &'a GrayImage,
    left: u32,
    top: u32,
}

impl NativeRefineBase<'_> {
    /// The composited pixel at a global canvas position, or `None` where the
    /// position is outside this window or not yet covered.
    fn masked(&self, gx: f64, gy: f64) -> Option<Rgb<f32>> {
        let lx = gx - f64::from(self.left);
        let ly = gy - f64::from(self.top);
        if lx < 0.0 || ly < 0.0 {
            return None;
        }
        let pixel = rgb_at(self.image, lx, ly)?;
        (self.mask.get_pixel(lx as u32, ly as u32)[0] != 0).then_some(pixel)
    }

    fn covered(&self, gx: u32, gy: u32) -> bool {
        let Some(lx) = gx.checked_sub(self.left) else {
            return false;
        };
        let Some(ly) = gy.checked_sub(self.top) else {
            return false;
        };
        lx < self.mask.width() && ly < self.mask.height() && self.mask.get_pixel(lx, ly)[0] != 0
    }
}

/// Observations of one spacing pass.
///
/// Kept apart from the pass that applies them because the streaming path fills
/// one accumulator from several tiles before the displacement field is built.
struct NativeRefinePass {
    observations: Vec<(Point2<f64>, [f64; 2], f64)>,
    accepted_grid: intra_station::ControlPointGrid,
    global_errors: Vec<f64>,
    local_errors: Vec<f64>,
    evaluated: usize,
    valid_cells: usize,
    max_round_trip: f64,
    similarity: SimilarityStats,
}

impl NativeRefinePass {
    fn new(width: u32, height: u32) -> Self {
        let columns = (width.saturating_sub(40) as usize / NATIVE_REFINE_STEP as usize) + 1;
        let rows = (height.saturating_sub(40) as usize / NATIVE_REFINE_STEP as usize) + 1;
        Self {
            observations: Vec::new(),
            accepted_grid: intra_station::ControlPointGrid::new(columns, rows),
            global_errors: Vec::new(),
            local_errors: Vec::new(),
            evaluated: 0,
            valid_cells: 0,
            max_round_trip: 0.0,
            similarity: SimilarityStats::default(),
        }
    }
}

/// Width of one correlation bucket, and the value the lowest bucket starts at.
const SIMILARITY_BUCKET_WIDTH: f64 = 0.05;
const SIMILARITY_BUCKET_FLOOR: f64 = 0.50;
const SIMILARITY_BUCKETS: usize = 10;

/// Correlation the round-trip distribution is reported from, so the log states
/// the tradeoff at the similarity floor that is actually in force.
const SIMILARITY_REPORTED_FLOOR: f64 = intra_station::INTRA_STATION_MIN_CORRELATION;

/// Which correlation bucket a score belongs to, or `None` below the floor.
fn similarity_bucket(correlation: f64) -> Option<usize> {
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
struct SimilarityStats {
    /// Forward peak correlation of every control point whose patch pair carried
    /// comparable signal at all.
    correlations: Vec<f64>,
    /// Per bucket, how many control points reached the reverse match.
    bucket_total: [usize; SIMILARITY_BUCKETS],
    /// Per bucket, how many of those closed the round trip.
    bucket_round_trip: [usize; SIMILARITY_BUCKETS],
    /// Round trip in native pixels of every control point that reached the
    /// reverse match at or above [`SIMILARITY_REPORTED_FLOOR`], so the 需求 2.4
    /// tolerance can be read off a distribution instead of guessed.
    accepted_round_trips: Vec<f64>,
    /// Second-peak margin of those same control points, so the margin gate can
    /// be checked against data too.
    accepted_peak_margins: Vec<f64>,
    /// `softer / sharper` high-frequency energy ratio of every measured pair,
    /// before and after the matched low-pass.
    focal_ratio_before: Vec<f64>,
    focal_ratio_matched: Vec<f64>,
    focal_ratio_after: Vec<f64>,
    /// Sigma, in patch samples, the focal-plane match decided the two sides
    /// differ by, and how often the descent ran out of budget.
    focal_passes: Vec<f64>,
    focal_capped: usize,
}

impl SimilarityStats {
    fn absorb(&mut self, point: &NativeControlPoint) {
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
    fn summary(&mut self) -> String {
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

    fn median_focal_ratios(&mut self) -> (f64, f64) {
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
struct NativeControlPoint {
    /// The cell centre lies on covered base, so its area counts towards the
    /// anchor frame's valid pixel area (需求 2.8).
    covered: bool,
    /// The patch pair was complete, so a match was attempted.
    evaluated: bool,
    /// Bidirectional round trip in native pixels, `0.0` when never measured.
    round_trip_px: f64,
    /// Forward peak correlation, `None` when the patch pair carried no
    /// comparable signal at all.
    best_correlation: Option<f64>,
    /// `true` when the forward match cleared the similarity gates, so a reverse
    /// match was run and the round trip is a real measurement.
    reverse_attempted: bool,
    /// `true` when that round trip closed inside the 需求 2.4 tolerance.
    round_trip_closed: bool,
    /// Gap between the correlation peak and the best hypothesis at least 2.5
    /// samples away, `None` when no hypothesis was scored.
    peak_margin: Option<f64>,
    /// What the focal-plane matched low-pass did to this pair.
    focal_plane: Option<intra_station::FocalPlaneMatch>,
    /// Displacement in analysis units, its correlation, the local (round trip)
    /// and global symmetric reprojection errors in native pixels.  `Some` as soon
    /// as both directions matched, i.e. *before* the 需求 2.4 and 需求 2.6 gates.
    ///
    /// Published separately from [`Self::accepted`] so the gates can be stated as
    /// a filter over a measurement rather than as the measurement itself.
    measured: Option<([f64; 2], f64, f64, f64)>,
    /// [`Self::measured`] once the 需求 2.4 and 需求 2.6 gates have passed.
    accepted: Option<([f64; 2], f64, f64, f64)>,
}

