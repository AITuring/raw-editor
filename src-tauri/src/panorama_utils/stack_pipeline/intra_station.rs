//! Intra_Station_Registrar: anchor frame selection, the two-level registration
//! budget and the control-point gates of 需求 2.1–2.9.
//!
//! The registrar itself is not a new parallel pipeline: the measurement work
//! lives in [`super::super::mosaic::refine_native_layer`] and
//! [`super::super::registration::refine_warped_patch`].  This module owns the
//! *decisions* — which frame is the anchor, at which resolution the two global
//! and local rounds run, which control point survives, and when a whole frame
//! falls back to its global model — so each rule is a small deterministic
//! function a property test can drive without decoding a RAW file.

use std::sync::Mutex;

use image::{GrayImage, Rgb};

use super::super::registration::PatchMatchGates;
use super::report::{
    IntraStationFrameRecord, IntraStationFrameStatus, IntraStationLocalFieldVerdict,
    IntraStationReport,
};

/// Actual configured long side of the two intra-station analysis rounds.
///
/// This is deliberately *not* the requirement ceiling: Stack_Report records
/// the value the production path used, while
/// [`INTRA_STATION_ANALYSIS_MAX_LONG_SIDE`] keeps that configuration bounded
/// (需求 2.2). It is also separate from `scalable_alignment_budget()`'s
/// 2400/1800/1536 inter-station budget.
pub(crate) const INTRA_STATION_ANALYSIS_LONG_SIDE: u32 = 1_400;

/// Hard upper bound for the actual intra-station analysis long side (需求 2.2).
pub(crate) const INTRA_STATION_ANALYSIS_MAX_LONG_SIDE: u32 = 2_048;

/// Finite native-pixel ceiling for neighbouring refinement controls (需求 2.3).
pub(crate) const INTRA_STATION_CONTROL_POINT_MAX_SPACING_PX: f64 = 112.0;

/// Maximum refinement-grid spacing relative to the ownership cell side.
///
/// The refinement grid is intentionally coarser than the ownership grid: it
/// estimates a smooth geometric field, while ownership routes pixel source
/// labels around detail. Bounding the ratio prevents the former from becoming
/// arbitrarily sparse without claiming both grids have equal density.
pub(crate) const INTRA_STATION_CONTROL_POINT_MAX_OWNERSHIP_RATIO: f64 = 8.0;

/// Whether a measured native control-point spacing obeys the bounded feasible
/// rule `s <= min(112px, 8c)` (需求 2.3).
pub(crate) fn control_point_spacing_is_bounded(spacing_px: f64, ownership_cell_px: u32) -> bool {
    spacing_px.is_finite()
        && spacing_px >= 0.0
        && ownership_cell_px > 0
        && spacing_px <= INTRA_STATION_CONTROL_POINT_MAX_SPACING_PX
        && spacing_px
            <= INTRA_STATION_CONTROL_POINT_MAX_OWNERSHIP_RATIO * f64::from(ownership_cell_px)
}

/// Number of global + local rounds at [`INTRA_STATION_ANALYSIS_LONG_SIDE`]
/// before the native patch refinement runs (需求 2.2).
pub(crate) const INTRA_STATION_ANALYSIS_ROUNDS: usize = 2;

/// Upper bound of the native patch search reach, in native pixels of the anchor
/// frame (需求 2.3 allows 64, and this uses all of it).
///
/// Separate from `FOCUS_MATCH_REFINE_SEARCH_RADIUS = 24`, because focus
/// breathing displaces a frame further than an inter-station residual does.
///
/// This is a *bound*, not an argument: `refine_warped_patch`'s two integer
/// parameters are the matching window radius and the translation search extent,
/// both counted in patch samples, and both have to fit inside the patch buffer
/// together with the whole search support.  Passing 48 as the window radius of a
/// 57×57 buffer asks the function for samples at −20, which makes it return
/// `None` for *every* control point — a silent, total disabling of the native
/// refinement rather than a wider search.  [`native_search_reach_px`] converts
/// the two sample-space parameters into the native reach this bound governs.
pub(crate) const INTRA_STATION_SEARCH_RADIUS: f64 = 64.0;

/// Matching window radius in patch samples.
///
/// 需求 2.3 asks for a window side of at least 32 native pixels and a search
/// reach of at most 64.  At the finest `spacing = 1.0` pass this radius gives a
/// window side of `2 * 25 + 1 = 51` native pixels, and together with
/// [`INTRA_STATION_SEARCH_SAMPLES`] it puts the coarse `spacing = 2.0` reach at
/// exactly the 64 the requirement allows.
///
/// The smallest radius that clears the requirement is 16, for a 33 pixel window,
/// and that is what this used to be.  On the frozen fixtures a 33 pixel window
/// on sparse ink over blank silk carried too little structure for the
/// correlation to separate a match from grain: the correlation median over one
/// station's control points was 0.60 against 0.84 on a densely painted station.
/// A 51 pixel window covers 2.4 times the area, and the correlation of a true
/// match rises with the square root of the structure inside it.
pub(crate) const INTRA_STATION_MATCH_RADIUS: i32 = 25;

/// Translation search extent in patch samples.  The reach in native pixels is
/// `(INTRA_STATION_MATCH_RADIUS + INTRA_STATION_SEARCH_SAMPLES) * spacing`,
/// which the coarse `spacing = 2.0` pass turns into exactly the 64 native pixels
/// 需求 2.3 allows — and which still fits the patch buffer.
pub(crate) const INTRA_STATION_SEARCH_SAMPLES: i32 = 7;

/// Half side of the patch buffers handed to the matcher on the intra-station
/// path (67x67).  The legacy mosaic path keeps its own 57x57.
///
/// One sample wider than `INTRA_STATION_MATCH_RADIUS + INTRA_STATION_SEARCH_SAMPLES`
/// because bilinear sampling needs a neighbour: the matcher reserves the whole
/// search support before it scores a hypothesis, and a support that lands exactly
/// on the last row makes it decline every control point.
pub(crate) const INTRA_STATION_PATCH_HALF: i32 = 33;

/// Half side of the legacy mosaic path's patch buffers.  Unchanged, because the
/// comparison path's output has to stay byte identical.
pub(crate) const LEGACY_MOSAIC_PATCH_HALF: i32 = 28;

/// Native-pixel reach of one refinement pass at `spacing` native pixels per
/// patch sample (需求 2.3).
pub(crate) fn native_search_reach_px(spacing: f64) -> f64 {
    f64::from(INTRA_STATION_MATCH_RADIUS + INTRA_STATION_SEARCH_SAMPLES) * spacing
}

/// `true` when the matching window plus the whole search support fits inside the
/// patch buffer, which is what `refine_warped_patch` requires before it will
/// score a single hypothesis.
pub(crate) fn patch_support_fits(match_radius: i32, search_samples: i32, patch_half: i32) -> bool {
    // Strictly less than: bilinear sampling of the outermost support position
    // needs the next sample too.
    match_radius > 0 && search_samples > 0 && match_radius + search_samples < patch_half
}

// ---------------------------------------------------------------------------
// Focal-plane matched low-pass
// ---------------------------------------------------------------------------

/// Largest low-pass variance the focal-plane match may add to the sharper side
/// of a patch pair, in squared patch samples per axis.
///
/// A Capture_Station is a focus bracket, so at any one control point one frame
/// is near its focal plane and the other is not.  Defocus is a low-pass, and a
/// low-pass destroys the very band raw-intensity cross-correlation relies on:
/// measured on the frozen fixtures, matching a sharp 33x33 native window against
/// its own defocused counterpart scored below 0.80 almost everywhere, which is
/// why the acceptance rate collapsed to 0.02-0.4%.  Blurring the sharper side
/// back down to the softer side's scale restores a comparable band on both
/// sides, which is the whole point.
///
/// `16.0` is sigma 4 patch samples, i.e. 8 native pixels at the coarse
/// `spacing = 2.0` pass.  Past that the survivor is pure low frequency and the
/// subpixel peak would no longer be a pixel-scale measurement, so an unreachable
/// pair is better left unmatched than matched on a blur.
pub(crate) const INTRA_STATION_FOCAL_MATCH_MAX_VARIANCE: f64 = 16.0;

/// Finest low-pass variance step the descent will take.  Sigma 0.1 samples: any
/// finer and the step is below what the patch's own quantisation can resolve.
pub(crate) const INTRA_STATION_FOCAL_MATCH_MIN_STEP: f64 = 0.01;

/// Coarsest step, which is the `[1 2 1]/4` binomial and the largest variance a
/// non-negative three-tap kernel can carry.
pub(crate) const INTRA_STATION_FOCAL_MATCH_MAX_STEP: f64 = 0.5;

/// Step growth after an accepted step, so a deeply defocused pair is reached in
/// a logarithmic number of steps instead of a linear one.
pub(crate) const INTRA_STATION_FOCAL_MATCH_STEP_GROWTH: f64 = 1.6;

/// Step budget of the descent.  With the growth above this is enough to walk
/// from the minimum step to the whole variance budget and still bisect back.
pub(crate) const INTRA_STATION_FOCAL_MATCH_MAX_STEPS: usize = 64;

/// Baseline low-pass variance applied to *both* sides of every patch pair, in
/// squared patch samples per axis.  `0.5` is one `[1 2 1]/4` pass.
///
/// Sensor grain is independent between two exposures, so it is the one component
/// of a patch that can never correlate.  Taking the top octave off both sides
/// raises the correlation of a true match without moving the peak, because the
/// same linear filter is applied to both.
pub(crate) const INTRA_STATION_FOCAL_MATCH_BASE_VARIANCE: f64 = 0.5;

/// What [`match_focal_plane`] did to one patch pair.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub(crate) struct FocalPlaneMatch {
    /// Low-pass variance added to the sharper side, in squared patch samples per
    /// axis, excluding the shared baseline.  `sqrt` of it is the sigma the match
    /// decided the two focal planes differ by.
    pub(crate) matched_variance: f64,
    /// `softer / sharper` high-frequency energy before the match, in `[0, 1]`.
    /// `1.0` means the two sides were already at the same focal plane.
    pub(crate) sharpness_ratio_before: f64,
    /// The same ratio once the sharper side has been low-passed, before the
    /// shared baseline pass.  This is the number that says whether the match
    /// reached parity or ran out of budget.
    pub(crate) sharpness_ratio_matched: f64,
    /// The same ratio after the shared baseline pass, i.e. on the buffers the
    /// matcher actually correlates.
    pub(crate) sharpness_ratio_after: f64,
    /// `true` when the descent spent its whole variance budget without reaching
    /// parity, so the two sides are still at different focal planes.
    pub(crate) capped: bool,
}

/// High-frequency energy of a patch: the mean squared second difference over the
/// interior, along both axes.
///
/// A second difference is blind to a gain or offset step and to any linear ramp,
/// so what it measures is the band defocus removes and nothing else.  This is
/// the same quantity the Glossary's Sharpness_Score is built from, reduced to
/// the single patch the matcher is about to score.
pub(crate) fn patch_high_frequency_energy(patch: &[f64], side: usize) -> f64 {
    if side < 3 || patch.len() < side * side {
        return 0.0;
    }
    let mut total = 0.0f64;
    let mut count = 0usize;
    for y in 1..side - 1 {
        for x in 1..side - 1 {
            let centre = patch[y * side + x] * 2.0;
            let horizontal = patch[y * side + x - 1] + patch[y * side + x + 1] - centre;
            let vertical = patch[(y - 1) * side + x] + patch[(y + 1) * side + x] - centre;
            total += horizontal * horizontal + vertical * vertical;
            count += 1;
        }
    }
    if count == 0 {
        0.0
    } else {
        total / count as f64
    }
}

/// Distance from parity between two high-frequency energies, in log space.
///
/// Log space, because one binomial pass divides the energy by a factor of three
/// to five: judged on the linear difference, overshooting from twice the target
/// to a fifth of it looks like an improvement, and it is not.
fn parity_distance(energy: f64, target: f64) -> f64 {
    const FLOOR: f64 = 1e-9;
    (energy.max(FLOOR).ln() - target.max(FLOOR).ln()).abs()
}

/// Low-pass `patch` towards `target_energy` and keep the amount of blur whose
/// high-frequency energy comes closest to it, returning the added variance in
/// squared patch samples per axis and whether the budget ran out first.
///
/// The step size adapts, because a fixed one cannot work at both ends.  A full
/// `[1 2 1]/4` pass adds variance 0.5 and divides the high-frequency energy by
/// three to five, which on the frozen fixtures overshot parity at the *first*
/// pass for the median control point: the pair typically starts at an energy
/// ratio near 0.7, and one pass takes it to roughly 0.2 on the far side of
/// parity.  Measured with a fixed pass the descent therefore kept zero passes
/// almost everywhere and the focal planes were never matched at all.  Growing an
/// accepted step and halving a rejected one lands on the amount of blur the pair
/// actually differs by, at both the sub-pass and the several-pixel end.
fn blur_towards_energy(
    patch: &mut [f64],
    side: usize,
    target_energy: f64,
    variance_budget: f64,
) -> (f64, bool) {
    let mut scratch = vec![0.0f64; side * side];
    let mut previous = patch.to_vec();
    let mut energy = patch_high_frequency_energy(patch, side);
    let mut distance = parity_distance(energy, target_energy);
    let mut step = INTRA_STATION_FOCAL_MATCH_MIN_STEP;
    let mut applied = 0.0f64;
    for _ in 0..INTRA_STATION_FOCAL_MATCH_MAX_STEPS {
        if energy <= target_energy {
            break;
        }
        let attempt = step.min(variance_budget - applied);
        if attempt < INTRA_STATION_FOCAL_MATCH_MIN_STEP {
            break;
        }
        previous.copy_from_slice(patch);
        blur_variance_in_place(patch, side, attempt, &mut scratch);
        let next_energy = patch_high_frequency_energy(patch, side);
        let next_distance = parity_distance(next_energy, target_energy);
        if next_distance < distance {
            energy = next_energy;
            distance = next_distance;
            applied += attempt;
            step = (attempt * INTRA_STATION_FOCAL_MATCH_STEP_GROWTH)
                .min(INTRA_STATION_FOCAL_MATCH_MAX_STEP);
        } else {
            // The step went past parity: put the patch back and try half of it.
            patch.copy_from_slice(&previous);
            step = attempt * 0.5;
        }
    }
    (
        applied,
        applied + INTRA_STATION_FOCAL_MATCH_MIN_STEP >= variance_budget,
    )
}

/// One separable three-tap low-pass of the given variance, with clamped edges.
///
/// The kernel is `[v/2, 1 - v, v/2]`, whose variance is exactly `v` per axis;
/// `v = 0.5` is the binomial `[1 2 1]/4`.  Keeping the variance as the parameter
/// is what lets the focal-plane match step by an amount rather than by a pass,
/// and repeated application simply adds variances.  `scratch` must be at least
/// `side * side` long and is overwritten.
pub(crate) fn blur_variance_in_place(
    patch: &mut [f64],
    side: usize,
    variance: f64,
    scratch: &mut [f64],
) {
    if side < 3 || patch.len() < side * side || scratch.len() < side * side {
        return;
    }
    let variance = variance.clamp(0.0, 0.5);
    if variance <= 0.0 {
        return;
    }
    let wing = variance * 0.5;
    let centre_weight = 1.0 - variance;
    let at = |values: &[f64], x: usize, y: usize| values[y * side + x];
    for y in 0..side {
        for x in 0..side {
            let left = at(patch, x.saturating_sub(1), y);
            let right = at(patch, (x + 1).min(side - 1), y);
            scratch[y * side + x] = wing * (left + right) + centre_weight * at(patch, x, y);
        }
    }
    for y in 0..side {
        for x in 0..side {
            let up = at(scratch, x, y.saturating_sub(1));
            let down = at(scratch, x, (y + 1).min(side - 1));
            patch[y * side + x] = wing * (up + down) + centre_weight * at(scratch, x, y);
        }
    }
}

/// Bring two patches of the same content to a comparable focal plane (需求 2.3).
///
/// The sharper side receives the number of binomial low-pass passes that brings
/// its high-frequency energy closest to the softer side's, then both sides
/// receive the same baseline pass.  The filter is identical on both sides at
/// every step where both receive it, and the extra passes only *remove* signal
/// from the side that
/// has more of it, so the operation cannot manufacture agreement between two
/// different scenes: a patch pair that does not depict the same content ends up
/// with two low-passed but still uncorrelated signals, and the correlation gate
/// still rejects it.
pub(crate) fn match_focal_plane(a: &mut [f64], b: &mut [f64], side: usize) -> FocalPlaneMatch {
    let mut scratch = vec![0.0f64; side * side];
    let mut energy_a = patch_high_frequency_energy(a, side);
    let mut energy_b = patch_high_frequency_energy(b, side);
    let ratio = |low: f64, high: f64| {
        if high > 0.0 {
            (low / high).clamp(0.0, 1.0)
        } else {
            1.0
        }
    };
    let mut outcome = FocalPlaneMatch {
        matched_variance: 0.0,
        sharpness_ratio_before: ratio(energy_a.min(energy_b), energy_a.max(energy_b)),
        sharpness_ratio_matched: 1.0,
        sharpness_ratio_after: 1.0,
        capped: false,
    };
    // The shared baseline low-pass comes first, so the energy match is solved on
    // the very buffers the matcher will correlate.  Doing it the other way round
    // was measured to undo the match: the same linear filter removes a different
    // share of the high-frequency energy from two differently shaped spectra, so
    // a pair brought to a ratio of 0.98 came back out at 0.55.
    blur_variance_in_place(
        a,
        side,
        INTRA_STATION_FOCAL_MATCH_BASE_VARIANCE,
        &mut scratch,
    );
    blur_variance_in_place(
        b,
        side,
        INTRA_STATION_FOCAL_MATCH_BASE_VARIANCE,
        &mut scratch,
    );
    energy_a = patch_high_frequency_energy(a, side);
    energy_b = patch_high_frequency_energy(b, side);
    outcome.sharpness_ratio_before = ratio(energy_a.min(energy_b), energy_a.max(energy_b));
    // Only the sharper side is blurred, and only towards the softer side's
    // scale.  Deciding which side that is once keeps the two from taking turns
    // and blurring each other away.
    if energy_a != energy_b {
        let (sharper, target) = if energy_a > energy_b {
            (&mut *a, energy_b)
        } else {
            (&mut *b, energy_a)
        };
        let (applied, capped) = blur_towards_energy(
            sharper,
            side,
            target,
            INTRA_STATION_FOCAL_MATCH_MAX_VARIANCE,
        );
        outcome.matched_variance = applied;
        outcome.capped = capped;
        energy_a = patch_high_frequency_energy(a, side);
        energy_b = patch_high_frequency_energy(b, side);
    }
    outcome.sharpness_ratio_matched = ratio(energy_a.min(energy_b), energy_a.max(energy_b));
    outcome.sharpness_ratio_after = outcome.sharpness_ratio_matched;
    outcome
}

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

/// Accepted control point displacements on the refinement grid (需求 2.5).
///
/// The grid is walked in row-major order, so the 8-neighbourhood a control
/// point is compared against is a deterministic function of the photographs.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ControlPointGrid {
    columns: usize,
    rows: usize,
    accepted: Vec<Option<[f64; 2]>>,
}

impl ControlPointGrid {
    pub(crate) fn new(columns: usize, rows: usize) -> Self {
        Self {
            columns,
            rows,
            accepted: vec![None; columns * rows],
        }
    }

    pub(crate) fn accept(&mut self, column: usize, row: usize, displacement: [f64; 2]) {
        if column < self.columns && row < self.rows {
            self.accepted[row * self.columns + column] = Some(displacement);
        }
    }

    pub(crate) fn accepted_count(&self) -> usize {
        self.accepted.iter().filter(|cell| cell.is_some()).count()
    }

    /// The accepted displacement of one control point, `None` when the gates
    /// rejected it or the grid does not hold that position.
    ///
    /// A query, for Property 6: the refinement itself only ever needs the
    /// neighbourhood median and the count.
    #[cfg(test)]
    pub(crate) fn accepted_at(&self, column: usize, row: usize) -> Option<[f64; 2]> {
        (column < self.columns && row < self.rows)
            .then(|| self.accepted[row * self.columns + column])
            .flatten()
    }

    /// Union this grid with another of the same shape.
    ///
    /// 需求 2.8 counts the control point cells that *contain* an accepted local
    /// match, and the refinement runs two spacing passes whose accepted matches
    /// both end up in the same displacement field.  A cell matched by the coarse
    /// pass therefore contains an accepted local match whether or not the fine
    /// pass matched it again, so the coverage is the union over passes and not
    /// whatever the last pass happened to see.
    ///
    /// The 需求 2.5 neighbourhood gate deliberately does *not* use the union: a
    /// coarse-pass displacement has already been applied to the field, so the
    /// fine pass measures what is left and the two are not comparable.
    pub(crate) fn absorb(&mut self, other: &Self) {
        if self.columns != other.columns || self.rows != other.rows {
            return;
        }
        for (cell, incoming) in self.accepted.iter_mut().zip(other.accepted.iter()) {
            if cell.is_none() {
                *cell = *incoming;
            }
        }
    }

    /// Per-axis median of the accepted 8-neighbourhood, or `None` when no
    /// neighbour has been accepted yet.
    pub(crate) fn neighbour_median(&self, column: usize, row: usize) -> Option<[f64; 2]> {
        let mut xs = Vec::with_capacity(8);
        let mut ys = Vec::with_capacity(8);
        for row_offset in -1i64..=1 {
            for column_offset in -1i64..=1 {
                if row_offset == 0 && column_offset == 0 {
                    continue;
                }
                let neighbour_row = row as i64 + row_offset;
                let neighbour_column = column as i64 + column_offset;
                if neighbour_row < 0
                    || neighbour_column < 0
                    || neighbour_row >= self.rows as i64
                    || neighbour_column >= self.columns as i64
                {
                    continue;
                }
                if let Some(displacement) =
                    self.accepted[neighbour_row as usize * self.columns + neighbour_column as usize]
                {
                    xs.push(displacement[0]);
                    ys.push(displacement[1]);
                }
            }
        }
        (!xs.is_empty()).then(|| [median_of(&mut xs), median_of(&mut ys)])
    }
}

/// Deterministic median: `total_cmp` ordering, and the lower of the two middle
/// samples for an even count, so the value never depends on sort stability.
pub(crate) fn median_of(values: &mut [f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(|left, right| left.total_cmp(right));
    values[(values.len() - 1) / 2]
}

// ---------------------------------------------------------------------------
// Run scoped observations
// ---------------------------------------------------------------------------

/// Intra-station records of the run currently in flight.
///
/// The refinement sites are free functions deep inside `mosaic.rs` with no
/// report parameter, exactly like the degradation observation points, so the
/// records are collected in a run scoped global and folded into the report by
/// [`super::report::StackReportRecorder::write_once`].
static RUN_INTRA_STATION: Mutex<Vec<IntraStationReport>> = Mutex::new(Vec::new());

fn with_run_records<T>(body: impl FnOnce(&mut Vec<IntraStationReport>) -> T) -> T {
    let mut guard = RUN_INTRA_STATION
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    body(&mut guard)
}

/// Starts a fresh observation window, once per stitching run.
pub(crate) fn reset_run_records() {
    with_run_records(Vec::clear);
}

/// Add one frame record to a per-station collection (需求 2.9).  Frames keep
/// the order in which they were first registered and a re-registered path
/// replaces its record; the station entry is created on first use.  Separated
/// from the global sink so it is testable on its own.
pub(crate) fn push_frame_record(
    records: &mut Vec<IntraStationReport>,
    station_index: usize,
    anchor_path: &str,
    record: IntraStationFrameRecord,
) {
    let station = match records
        .iter_mut()
        .find(|station| station.station_index == station_index)
    {
        Some(station) => station,
        None => {
            records.push(IntraStationReport {
                station_index,
                anchor_path: anchor_path.to_string(),
                analysis_long_side: INTRA_STATION_ANALYSIS_LONG_SIDE,
                frames: Vec::new(),
            });
            records.last_mut().expect("just pushed")
        }
    };
    if station.anchor_path.is_empty() {
        station.anchor_path = anchor_path.to_string();
    }
    // A station is rendered more than once per run (station matching, a
    // rendering-prior repair, composition). Each render re-measures its frames,
    // so the latest measurement replaces the frame's earlier record in place
    // instead of listing the same Source_RAW twice in Stack_Report.
    match station
        .frames
        .iter_mut()
        .find(|frame| frame.path == record.path)
    {
        Some(existing) => *existing = record,
        None => station.frames.push(record),
    }
}

/// Record one frame of one Capture_Station of the current run (需求 2.9).
pub(crate) fn record_run_frame(
    station_index: usize,
    anchor_path: &str,
    record: IntraStationFrameRecord,
) {
    with_run_records(|records| push_frame_record(records, station_index, anchor_path, record))
}

/// Copy of the records for report serialisation.  Stations are ordered by
/// index so the document does not inherit the registration order.
pub(crate) fn run_records_snapshot() -> Vec<IntraStationReport> {
    let mut records = with_run_records(|records| records.clone());
    records.sort_by_key(|station| station.station_index);
    records
}

#[cfg(test)]
mod tests;
