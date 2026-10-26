// Intra_Station_Registrar fragment; included at parent module scope.

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

