//! Native-resolution residual refinement passes (需求 2.3–2.9).
//!
//! Refinement modes and outcomes, per-pass bookkeeping, folding accepted
//! observations into the displacement field, the per-frame verdict and the
//! in-memory and streaming entry points shared by both compositors.

use super::super::registration::PatchMatchGates;
use super::super::stack_pipeline::degradation;
use super::super::stack_pipeline::intra_station;
use super::super::stack_pipeline::report::IntraStationFrameRecord;
use super::super::stack_pipeline::report::IntraStationFrameStatus;
use image::GrayImage;
use image::Luma;
use image::Rgb;
use image::Rgb32FImage;
use nalgebra::Matrix3;
use nalgebra::Point2;

use super::{
    Field, LayerSampler, NATIVE_FIELD_RADIUS, NATIVE_REFINE_STEP, SimilarityStats,
    StreamingMosaicStore, collect_native_observations, median, rgb_at,
};

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
    pub(super) fn patch_geometry(self) -> (i32, i32) {
        match self {
            Self::LegacyMosaic => (10, 7),
            Self::IntraStation { .. } => (
                intra_station::INTRA_STATION_MATCH_RADIUS,
                intra_station::INTRA_STATION_SEARCH_SAMPLES,
            ),
        }
    }

    pub(super) fn anchor_long_side(self) -> Option<u32> {
        match self {
            Self::LegacyMosaic => None,
            Self::IntraStation { anchor_long_side } => Some(anchor_long_side),
        }
    }

    /// Half side of the patch buffers this mode builds.
    pub(super) fn patch_half(self) -> i32 {
        match self {
            Self::LegacyMosaic => intra_station::LEGACY_MOSAIC_PATCH_HALF,
            Self::IntraStation { .. } => intra_station::INTRA_STATION_PATCH_HALF,
        }
    }

    /// Similarity gates of one control point.
    pub(super) fn patch_gates(self) -> PatchMatchGates {
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
    pub(super) fn matches_focal_plane(self) -> bool {
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
pub(super) fn intra_station_frame_record(
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
pub(super) struct NativeRefineBase<'a> {
    pub(super) image: &'a Rgb32FImage,
    pub(super) mask: &'a GrayImage,
    pub(super) left: u32,
    pub(super) top: u32,
}

impl NativeRefineBase<'_> {
    /// The composited pixel at a global canvas position, or `None` where the
    /// position is outside this window or not yet covered.
    pub(super) fn masked(&self, gx: f64, gy: f64) -> Option<Rgb<f32>> {
        let lx = gx - f64::from(self.left);
        let ly = gy - f64::from(self.top);
        if lx < 0.0 || ly < 0.0 {
            return None;
        }
        let pixel = rgb_at(self.image, lx, ly)?;
        (self.mask.get_pixel(lx as u32, ly as u32)[0] != 0).then_some(pixel)
    }

    pub(super) fn covered(&self, gx: u32, gy: u32) -> bool {
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
pub(super) struct NativeRefinePass {
    pub(super) observations: Vec<(Point2<f64>, [f64; 2], f64)>,
    pub(super) accepted_grid: intra_station::ControlPointGrid,
    pub(super) global_errors: Vec<f64>,
    pub(super) local_errors: Vec<f64>,
    pub(super) evaluated: usize,
    pub(super) valid_cells: usize,
    pub(super) max_round_trip: f64,
    pub(super) similarity: SimilarityStats,
}

impl NativeRefinePass {
    pub(super) fn new(width: u32, height: u32) -> Self {
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

/// Fold one spacing pass into the outcome and, when it carries enough accepted
/// control points, into the sampler's displacement field.
#[allow(clippy::too_many_arguments)]
pub(super) fn apply_native_pass(
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
pub(super) fn finalize_native_refinement(
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

/// Native patch refinement against an in-memory canvas (需求 2.3–2.8).
pub(super) fn refine_native_layer(
    base: &Rgb32FImage,
    base_mask: &GrayImage,
    sampler: &mut LayerSampler<'_>,
    width: u32,
    height: u32,
    mode: NativeRefineMode,
) -> NativeRefineOutcome {
    let mut outcome = NativeRefineOutcome::default();
    if mode == NativeRefineMode::LegacyMosaic && sampler.scale <= 2.0 {
        return outcome;
    }
    outcome.skipped = false;
    outcome.control_point_spacing_px = native_control_point_spacing_px(sampler.scale);
    let entry_field = sampler.residual.clone();
    let cell_area_px = (NATIVE_REFINE_STEP as f64 * sampler.scale).powi(2);
    let mut accepted_cells = NativeRefinePass::new(width, height).accepted_grid;
    let base = NativeRefineBase {
        image: base,
        mask: base_mask,
        left: 0,
        top: 0,
    };
    for (index, spacing) in [2.0, 1.0].into_iter().enumerate() {
        let mut pass = NativeRefinePass::new(width, height);
        collect_native_observations(&base, sampler, width, height, spacing, mode, &mut pass);
        apply_native_pass(
            pass,
            sampler,
            &mut outcome,
            &mut accepted_cells,
            cell_area_px,
            spacing,
            index == 0,
            mode,
        );
    }
    finalize_native_refinement(&mut outcome, sampler, entry_field, mode);
    outcome
}

/// Native patch refinement against a tiled canvas (需求 2.3–2.8).
///
/// This is the same measurement as [`refine_native_layer`]; only the base pixels
/// arrive one tile at a time, because the streaming compositor is used exactly
/// when the canvas is too large to hold.  Without this entry point the whole of
/// 需求 2.3–2.9 would be unreachable for every multi-station stack, which is the
/// only kind of stack the streaming compositor ever sees.
#[allow(clippy::too_many_arguments)]
pub(super) fn refine_native_layer_streaming(
    store: &StreamingMosaicStore,
    owner: Option<u8>,
    sampler: &mut LayerSampler<'_>,
    width: u32,
    height: u32,
    layer: (u32, u32, u32, u32),
    mode: NativeRefineMode,
) -> Result<NativeRefineOutcome, String> {
    let mut outcome = NativeRefineOutcome {
        skipped: false,
        control_point_spacing_px: native_control_point_spacing_px(sampler.scale),
        ..Default::default()
    };
    let entry_field = sampler.residual.clone();
    let cell_area_px = (NATIVE_REFINE_STEP as f64 * sampler.scale).powi(2);
    let mut accepted_cells = NativeRefinePass::new(width, height).accepted_grid;
    let (left, right, top, bottom) = layer;
    for (index, spacing) in [2.0, 1.0].into_iter().enumerate() {
        let mut pass = NativeRefinePass::new(width, height);
        for tile_row in top / store.tile_size..=bottom / store.tile_size {
            for tile_column in left / store.tile_size..=right / store.tile_size {
                let (tile_left, tile_top, tile_width, tile_height) =
                    store.tile_extent(tile_column, tile_row);
                let (tile, covered) = store.load_tile(tile_column, tile_row)?;
                // A later focal plane of a station must register against its own
                // camera position: pixels owned by another station are available
                // for colour matching but never for residual motion.
                let tile_mask = match owner {
                    Some(group) => {
                        let tile_owner = store.load_owner_tile(tile_column, tile_row);
                        GrayImage::from_fn(tile_width, tile_height, |x, y| {
                            Luma([
                                if covered.get_pixel(x, y)[0] != 0
                                    && tile_owner.get_pixel(x, y)[0] == group
                                {
                                    255
                                } else {
                                    0
                                },
                            ])
                        })
                    }
                    None => covered,
                };
                if tile_mask.as_raw().iter().all(|&value| value == 0) {
                    continue;
                }
                let base = NativeRefineBase {
                    image: &tile,
                    mask: &tile_mask,
                    left: tile_left,
                    top: tile_top,
                };
                collect_native_observations(
                    &base, sampler, width, height, spacing, mode, &mut pass,
                );
            }
        }
        apply_native_pass(
            pass,
            sampler,
            &mut outcome,
            &mut accepted_cells,
            cell_area_px,
            spacing,
            index == 0,
            mode,
        );
    }
    finalize_native_refinement(&mut outcome, sampler, entry_field, mode);
    Ok(outcome)
}

/// Distance between neighbouring control points in native pixels (需求 2.3).
///
/// [`NATIVE_REFINE_STEP`] is a step in *analysis* units, so the native spacing
/// is that step times the layer's analysis scale.  Published so the default
/// station path can compare the measured spacing against the Focus_Fuser's
/// ownership cell side.
pub(crate) fn native_control_point_spacing_px(analysis_scale: f64) -> f64 {
    NATIVE_REFINE_STEP as f64 * analysis_scale
}
