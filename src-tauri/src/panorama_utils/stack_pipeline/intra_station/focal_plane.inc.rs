// Intra_Station_Registrar fragment; included at parent module scope.

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

