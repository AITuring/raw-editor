// Intra_Station_Registrar fragment; included at parent module scope.

/// Bidirectional consistency limit in native pixels.  需求 2.4 allows 1.0; the
/// existing native refinement already held 0.40 and keeping the stricter value
/// is allowed (it only rejects more matches).  The measured value goes into the
/// report so the acceptance harness can see the real distribution.
pub(crate) const INTRA_STATION_BIDIRECTIONAL_TOLERANCE_PX: f64 = 1.00;

/// Neighbourhood consistency limit in native pixels (需求 2.5).
pub(crate) const INTRA_STATION_NEIGHBOUR_MEDIAN_TOLERANCE_PX: f64 = 4.0;

/// Symmetric reprojection error limit as a fraction of the anchor frame's
/// native long side (需求 2.6).
pub(crate) const INTRA_STATION_SYMMETRIC_ERROR_RATIO: f64 = 0.01;

/// Minimum inlier spatial coverage of a frame (需求 2.8).
pub(crate) const INTRA_STATION_MIN_INLIER_AREA_COVERAGE: f64 = 0.20;

/// Similarity gates of an intra-station control point.
///
/// See [`PatchMatchGates`] for why these are not the mosaic path's numbers, and
/// the module tests for the measured accept-rate/precision tradeoff these values
/// sit on.
pub(crate) const INTRA_STATION_PATCH_GATES: PatchMatchGates = PatchMatchGates {
    min_correlation: INTRA_STATION_MIN_CORRELATION,
    min_peak_margin: INTRA_STATION_MIN_PEAK_MARGIN,
    min_source_variance: 2.0,
    min_target_variance: 1.0,
};

/// Smallest correlation an intra-station control point may be accepted at.
///
/// Measured, on the frozen 10-frame fixtures, as the knee of the round-trip
/// precision curve.  Every refinement pass logs the share of each 0.05
/// correlation bucket that then closed the 需求 2.4 bidirectional round trip, and
/// pooled over the seven non-anchor frames of `langyuan-10` the fine pass gave:
///
/// | correlation | round trip closed |
/// |-------------|-------------------|
/// | 0.50        | 40%               |
/// | 0.55        | 51%               |
/// | 0.60        | 61%               |
/// | 0.65        | 69%               |
/// | 0.70        | 79%               |
/// | 0.75        | 86%               |
/// | 0.80        | 91%               |
/// | 0.85        | 93%               |
/// | 0.90        | 87%               |
/// | 0.95        | 73%               |
///
/// 0.70 is the lowest bucket still above three quarters, and the step down to
/// 0.65 is the largest in the table.  Below it more than a third of what would be
/// accepted is bidirectionally inconsistent, i.e. not a located match.
///
/// The curve also *falls* above 0.85, which is why the floor is not pushed
/// higher: after the matched low-pass a featureless patch correlates near 1.0 at
/// every hypothesis, so a very high score is evidence of a flat score surface
/// rather than of a well-located match — and the round trip catches it.  The
/// previous implementation's second, undocumented floor of 0.90 was sitting in
/// exactly that region.
pub(crate) const INTRA_STATION_MIN_CORRELATION: f64 = 0.70;

/// Smallest gap between the correlation peak and the best hypothesis at least
/// 2.5 samples away.
///
/// Unchanged from the value the patch matcher has always used, because nothing
/// measured here argues for loosening it: it is the guard that separates a
/// located stroke from the repeated ruling of a paper substrate, and the
/// bidirectional round trip cross-checks the same thing.
pub(crate) const INTRA_STATION_MIN_PEAK_MARGIN: f64 = 0.006;

/// Smallest number of accepted control points a median may be formed from
/// before 需求 2.7 is allowed to compare two of them.
///
/// The same threshold the displacement field itself requires: a pass with fewer
/// accepted points does not modify the field, so it also cannot describe the
/// frame's error.
pub(crate) const INTRA_STATION_MIN_BASELINE_SAMPLES: usize = 6;

/// Two anchor candidates within this Sharpness_Score distance are a tie and are
/// resolved by ascending absolute path (需求 2.1).
pub(crate) const INTRA_STATION_ANCHOR_SHARPNESS_TIE: f64 = 0.01;

/// Probe grid used to summarise a frame's Sharpness_Score.  A fixed count keeps
/// the cost independent of the source resolution and the result independent of
/// how the analysis image happened to be scaled.
const ANCHOR_PROBE_GRID: u32 = 48;

/// One frame competing for the anchor role of a Capture_Station (需求 2.1).
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct AnchorCandidate {
    /// Absolute path, which is also the tie-break key.
    pub path: String,
    /// Median Sharpness_Score over the frame's valid pixels.
    pub median_sharpness: f64,
}

/// Pick the anchor frame of a Capture_Station (需求 2.1).
///
/// The highest median Sharpness_Score wins.  Every candidate whose score is
/// less than [`INTRA_STATION_ANCHOR_SHARPNESS_TIE`] below the best is a tie and
/// the lowest absolute path among those wins, so the choice never depends on
/// import order, and two runs over the same photographs agree bit for bit.
///
/// `median_sharpness` may be either the raw acutance ratio or the normalised
/// Sharpness_Score of [`super::focus_fuser::normalized_sharpness`]: that mapping
/// is strictly increasing, so it cannot change which candidate is *highest* —
/// only which near-equal candidates fall inside the tie window.
pub(crate) fn select_anchor(candidates: &[AnchorCandidate]) -> Option<usize> {
    let best = candidates
        .iter()
        .map(|candidate| candidate.median_sharpness)
        .fold(f64::NEG_INFINITY, f64::max);
    if !best.is_finite() {
        // No candidate produced a measurable score: fall back to the lowest
        // absolute path, which is still content independent of import order.
        return candidates
            .iter()
            .enumerate()
            .min_by(|(left_index, left), (right_index, right)| {
                left.path
                    .cmp(&right.path)
                    .then_with(|| left_index.cmp(right_index))
            })
            .map(|(index, _)| index);
    }
    candidates
        .iter()
        .enumerate()
        .filter(|(_, candidate)| {
            best - candidate.median_sharpness < INTRA_STATION_ANCHOR_SHARPNESS_TIE
        })
        .min_by(|(left_index, left), (right_index, right)| {
            left.path
                .cmp(&right.path)
                .then_with(|| left_index.cmp(right_index))
        })
        .map(|(index, _)| index)
}

/// Probe spacing that keeps a measurement at or below
/// [`INTRA_STATION_ANALYSIS_LONG_SIDE`] (需求 2.2).  `1.0` means the analysis
/// image is already at or below the budget.
pub(crate) fn analysis_probe_spacing(long_side: u32) -> f64 {
    (f64::from(long_side) / f64::from(INTRA_STATION_ANALYSIS_LONG_SIDE)).max(1.0)
}

/// Median Sharpness_Score over the valid pixels of an analysis image (需求 2.1).
///
/// `measure` is the Sharpness_Score probe; it returns `None` where the window
/// would leave the valid pixels.  Probes are laid out on a fixed grid so the
/// median is a property of the photograph, not of the sampling density.
pub(crate) fn median_valid_sharpness<F>(width: u32, height: u32, mut measure: F) -> f64
where
    F: FnMut(f64, f64) -> Option<f64>,
{
    if width == 0 || height == 0 {
        return 0.0;
    }
    let spacing = analysis_probe_spacing(width.max(height));
    // The acutance window spans ±6 probe steps; stay inside it.
    let margin = 6.0 * spacing + 1.0;
    let usable_width = f64::from(width) - 2.0 * margin;
    let usable_height = f64::from(height) - 2.0 * margin;
    if usable_width <= 0.0 || usable_height <= 0.0 {
        return 0.0;
    }
    let mut scores = Vec::with_capacity((ANCHOR_PROBE_GRID * ANCHOR_PROBE_GRID) as usize);
    for row in 0..ANCHOR_PROBE_GRID {
        for column in 0..ANCHOR_PROBE_GRID {
            let x =
                margin + (f64::from(column) + 0.5) / f64::from(ANCHOR_PROBE_GRID) * usable_width;
            let y = margin + (f64::from(row) + 0.5) / f64::from(ANCHOR_PROBE_GRID) * usable_height;
            if let Some(score) = measure(x, y) {
                scores.push(score);
            }
        }
    }
    median_of(&mut scores)
}

/// Bilinear gray sample published as an RGB triple, so a gray analysis image
/// can feed the same Sharpness_Score probe the colour path uses.
pub(crate) fn gray_sample(image: &GrayImage, x: f64, y: f64) -> Option<Rgb<f32>> {
    if !x.is_finite() || !y.is_finite() || x < 0.0 || y < 0.0 {
        return None;
    }
    let x0 = x.floor();
    let y0 = y.floor();
    let x1 = x0 + 1.0;
    let y1 = y0 + 1.0;
    if x1 >= f64::from(image.width()) || y1 >= f64::from(image.height()) {
        return None;
    }
    let fx = x - x0;
    let fy = y - y0;
    let at = |px: f64, py: f64| f64::from(image.get_pixel(px as u32, py as u32)[0]) / 255.0;
    let top = at(x0, y0) * (1.0 - fx) + at(x1, y0) * fx;
    let bottom = at(x0, y1) * (1.0 - fx) + at(x1, y1) * fx;
    let value = (top * (1.0 - fy) + bottom * fy) as f32;
    Some(Rgb([value, value, value]))
}

/// Symmetric reprojection error ceiling of a control point (需求 2.6).
pub(crate) fn symmetric_error_limit_px(anchor_long_side: u32) -> f64 {
    f64::from(anchor_long_side) * INTRA_STATION_SYMMETRIC_ERROR_RATIO
}

/// 需求 2.4: the forward/backward round trip must close within the tolerance.
pub(crate) fn bidirectional_consistent(round_trip_px: f64) -> bool {
    round_trip_px.is_finite() && round_trip_px <= INTRA_STATION_BIDIRECTIONAL_TOLERANCE_PX
}

/// 需求 2.6: an implausible displacement claim keeps the global model at that
/// control point instead of bending the frame towards a repeated stroke.
pub(crate) fn symmetric_error_accepted(error_px: f64, anchor_long_side: u32) -> bool {
    error_px.is_finite() && error_px <= symmetric_error_limit_px(anchor_long_side)
}

/// 需求 2.5: agreement with the already accepted 8-neighbourhood.  A control
/// point with no accepted neighbour has nothing to disagree with and passes.
pub(crate) fn neighbour_median_consistent(
    displacement: [f64; 2],
    neighbour_median: Option<[f64; 2]>,
) -> bool {
    let Some(median) = neighbour_median else {
        return true;
    };
    (displacement[0] - median[0]).hypot(displacement[1] - median[1])
        <= INTRA_STATION_NEIGHBOUR_MEDIAN_TOLERANCE_PX
}

/// 需求 2.7: local refinement is kept only when it actually lowers the median
/// symmetric reprojection error of the frame.
pub(crate) fn local_field_improves(local_median_px: f64, global_median_px: f64) -> bool {
    local_median_px.is_finite()
        && global_median_px.is_finite()
        && local_median_px < global_median_px
}

/// 需求 2.7 evaluated against the evidence that actually exists.
///
/// The requirement reverts a frame whose refined median symmetric reprojection
/// error is *not below* the global model's.  That is a comparison between two
/// medians, and a median of an empty sample is not a value: taking it as `0.0`
/// turns the requirement into "revert unconditionally", which is what a coarse
/// pass that accepted too few control points used to cause.  Either side being
/// unformable is therefore its own outcome, distinct from both keeping and
/// reverting, and recorded as such in Stack_Report instead of silently reverting.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LocalFieldVerdict {
    /// The refinement lowered the frame's median error.
    Kept,
    /// The refinement did not lower it; 需求 2.7 reverts the frame.
    Reverted,
    /// One of the two medians has too few samples to exist, so the requirement's
    /// condition is unevaluable.  The measured field is kept, because reverting
    /// on absent evidence is a decision the requirement does not authorise.
    Indeterminate,
}

impl LocalFieldVerdict {
    /// `true` when the measured displacement field stays in effect.
    pub(crate) fn field_kept(self) -> bool {
        !matches!(self, Self::Reverted)
    }

    /// The stable machine-readable value written to Stack_Report (需求 2.9).
    pub(crate) fn record(self) -> IntraStationLocalFieldVerdict {
        match self {
            Self::Kept => IntraStationLocalFieldVerdict::Kept,
            Self::Reverted => IntraStationLocalFieldVerdict::Reverted,
            Self::Indeterminate => IntraStationLocalFieldVerdict::Indeterminate,
        }
    }
}

/// 需求 2.7 applied to one frame's two medians and their sample counts.
pub(crate) fn local_field_verdict(
    local_median_px: f64,
    local_samples: usize,
    global_median_px: f64,
    global_samples: usize,
) -> LocalFieldVerdict {
    if local_samples < INTRA_STATION_MIN_BASELINE_SAMPLES
        || global_samples < INTRA_STATION_MIN_BASELINE_SAMPLES
        || !local_median_px.is_finite()
        || !global_median_px.is_finite()
    {
        return LocalFieldVerdict::Indeterminate;
    }
    if local_field_improves(local_median_px, global_median_px) {
        LocalFieldVerdict::Kept
    } else {
        LocalFieldVerdict::Reverted
    }
}

/// 需求 2.8: accepted control point cell area over the anchor frame's valid
/// pixel area.  Both are measured in native pixels of the anchor frame.
pub(crate) fn inlier_area_coverage(
    accepted_cells: usize,
    cell_area_px: f64,
    anchor_valid_area_px: f64,
) -> f64 {
    let positive = |value: f64| value.is_finite() && value > 0.0;
    if !positive(anchor_valid_area_px) || !positive(cell_area_px) {
        return 0.0;
    }
    ((accepted_cells as f64 * cell_area_px) / anchor_valid_area_px).clamp(0.0, 1.0)
}

/// 需求 2.7 / 2.8: the per-frame verdict.
pub(crate) fn frame_status(
    local_field_kept: bool,
    inlier_area_coverage: f64,
) -> IntraStationFrameStatus {
    if inlier_area_coverage < INTRA_STATION_MIN_INLIER_AREA_COVERAGE {
        IntraStationFrameStatus::Failed
    } else if local_field_kept {
        IntraStationFrameStatus::Local
    } else {
        IntraStationFrameStatus::GlobalFallback
    }
}

