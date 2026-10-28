//! Tile_Compositor path selection (需求 15.1 / 15.2 / 15.3 / 15.10 / 15.11).
//!
//! The compositor a run uses is a **setting**, not an environment variable:
//! 需求 15.2 requires the layered Virtual_Tile path to be live in the default
//! build without any environment variable, command line argument or
//! configuration value being set, and 需求 15.10 requires every comparison
//! switch to be off in that same default build.  Both hold here because
//! [`StackCompositorChoice::default`] is [`StackCompositorChoice::LayeredVirtualTile`]
//! and an absent / unparsable setting resolves to that default.
//!
//! The layered path uses a bounded binary cut over real covered overlaps.

use super::report::{SelectedPath, StackReportRecorder};
use super::residual_warp::ResidualWarp;
use nalgebra::{Matrix3, Point2, Point3};

/// Compose the inverse mapping used by the Tile_Compositor.  A target world
/// coordinate is transformed to tile space once by the global inverse and
/// once by the residual inverse; the caller then performs one pixel sample at
/// the returned coordinate.  Keeping this as a pure function makes the
/// single-resample contract testable without decoding a Virtual_Tile.
/// Production samples through `map_target_to_source_with_residual`; this is
/// its test oracle.
#[cfg(test)]
pub(crate) fn tile_coordinate_from_world(
    world: Point2<f64>,
    tile_to_world_inverse: &Matrix3<f64>,
    residual_warp_inverse: &Matrix3<f64>,
) -> Option<Point2<f64>> {
    let tile = project_inverse(tile_to_world_inverse, world)?;
    // Requirement 8.7: tile_coord = warp_inverse(tile_to_world_inverse(world)).
    project_inverse(residual_warp_inverse, tile)
}

/// Resolve one source tile coordinate while applying the non-rigid residual
/// field. The global inverse is evaluated first to preserve the compositor's
/// one-coordinate path; the residual inverse is then queried in its native
/// world frame and converted back through that same inverse. With no matching
/// region (or an identity model) this is bit-for-bit the nominal coordinate.
pub(crate) fn tile_coordinate_with_residual(
    world: Point2<f64>,
    tile_to_world_inverse: &Matrix3<f64>,
    residual: &ResidualWarp,
    station_id: usize,
) -> Option<Point2<f64>> {
    let nominal = project_inverse(tile_to_world_inverse, world)?;
    if residual.regions.is_empty() && residual.line_layers.reports.is_empty() {
        return Some(nominal);
    }
    let source_world = residual.warp_inverse(world, station_id);
    if source_world == world {
        return Some(nominal);
    }
    project_inverse(tile_to_world_inverse, source_world)
}

fn project_inverse(matrix: &Matrix3<f64>, point: Point2<f64>) -> Option<Point2<f64>> {
    let projected = matrix * Point3::new(point.x, point.y, 1.0);
    if !projected.x.is_finite() || !projected.y.is_finite() || projected.z.abs() < 1e-12 {
        return None;
    }
    Some(Point2::new(
        projected.x / projected.z,
        projected.y / projected.z,
    ))
}

/// Ownership is a semantic plane and tone correction must not rewrite it.
/// Production hands the Tone_Harmonizer the ownership plane as an immutable
/// slice, so requirement 9.6 holds by construction; tests assert it.
#[cfg(test)]
pub(crate) fn assert_ownership_unchanged(before: &[u16], after: &[u16]) {
    assert_eq!(
        before.len(),
        after.len(),
        "tone changed ownership dimensions"
    );
    assert_eq!(before, after, "tone correction changed the Ownership_Map");
}

/// One authoritative path decision for both execution and Stack_Report.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct StackPathSelection {
    pub(crate) selected_path: SelectedPath,
    pub(crate) use_virtual_tiles: bool,
}

/// Which compositor composes the final canvas.
///
/// The identifiers match [`SelectedPath`] one for one so a Stack_Report can be
/// read back without a translation table.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) enum StackCompositorChoice {
    /// Default: layered Virtual_Tile composition with hard source ownership.
    #[default]
    LayeredVirtualTile,
    /// Comparison: the progressive minimum-cost seam tile compositor.
    ProgressiveSeamTile,
    /// Comparison: the streaming detail-preserving mosaic.
    StreamingMosaic,
    /// Comparison: the old single-layer `focus_stack_stitcher` (需求 15.3).
    LegacySingleLayerMosaic,
}

impl StackCompositorChoice {
    /// The stable machine readable identifier of this choice.
    pub(crate) fn as_identifier(self) -> &'static str {
        self.selected_path().as_identifier()
    }

    /// The Stack_Report identifier written to `report.selected_path` when this
    /// choice actually composes a multi-station run (需求 15.1 / 15.3).
    pub(crate) fn selected_path(self) -> SelectedPath {
        match self {
            Self::LayeredVirtualTile => SelectedPath::LayeredVirtualTile,
            Self::ProgressiveSeamTile => SelectedPath::ProgressiveSeamTile,
            Self::StreamingMosaic => SelectedPath::StreamingMosaic,
            Self::LegacySingleLayerMosaic => SelectedPath::LegacySingleLayerMosaic,
        }
    }

    /// `true` when the run is comparing against a retired path rather than
    /// composing the default one (需求 15.3 / 15.10).
    pub(crate) fn is_comparison_path(self) -> bool {
        !matches!(self, Self::LayeredVirtualTile)
    }

    /// Parse a persisted setting value.  Unknown values are rejected so a
    /// typo cannot silently disable the default path.
    pub(crate) fn from_identifier(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "layered_virtual_tile" | "layeredvirtualtile" => Some(Self::LayeredVirtualTile),
            "progressive_seam_tile" | "progressiveseamtile" => Some(Self::ProgressiveSeamTile),
            "streaming_mosaic" | "streamingmosaic" => Some(Self::StreamingMosaic),
            "legacy_single_layer_mosaic" | "legacysinglelayermosaic" => {
                Some(Self::LegacySingleLayerMosaic)
            }
            _ => None,
        }
    }

    /// Resolve the compositor of this run from the persisted setting.
    ///
    /// `None`, an empty string and an unrecognised value all resolve to the
    /// default layered path, which is what keeps 需求 15.2 (no external
    /// precondition for the default path) and 需求 15.10 (comparison switches
    /// off by default) true for a fresh install.
    ///
    /// The setting is the only source (任务 13.7): the two pre-setting
    /// environment switches are gone, so no environment variable can select
    /// or disable a compositor.
    pub(crate) fn resolve(setting: Option<&str>) -> Self {
        setting.and_then(Self::from_identifier).unwrap_or_default()
    }
}

/// Longest supported output canvas side in world native pixels (需求 10.2).
pub(crate) const MAX_OUTPUT_CANVAS_LONG_SIDE: u64 = 262_144;

/// The rejection of an output canvas whose long side exceeds
/// [`MAX_OUTPUT_CANVAS_LONG_SIDE`] (需求 10.11). The error starts with the
/// stable `canvas_long_side_exceeded` identifier and the degradation ledger
/// records the measured size and the limit; `None` means the canvas is
/// supported. Callers evaluate it before any canvas buffer is allocated.
pub(crate) fn reject_oversized_canvas(width: u64, height: u64) -> Option<String> {
    let long_side = width.max(height);
    if long_side <= MAX_OUTPUT_CANVAS_LONG_SIDE {
        return None;
    }
    super::degradation::record_run_degradation(
        super::degradation::CANVAS_LONG_SIDE_EXCEEDED,
        serde_json::json!({
            "width": width,
            "height": height,
            "long_side": long_side,
            "limit": MAX_OUTPUT_CANVAS_LONG_SIDE,
        }),
    );
    Some(format!(
        "{}: the output canvas {width}x{height} exceeds the supported long side of {} pixels",
        super::degradation::CANVAS_LONG_SIDE_EXCEEDED,
        MAX_OUTPUT_CANVAS_LONG_SIDE
    ))
}

/// `true` for an error produced by [`reject_oversized_canvas`]: a terminal
/// rejection that no fallback renderer may retry.
pub(crate) fn is_canvas_rejection(error: &str) -> bool {
    error.starts_with(super::degradation::CANVAS_LONG_SIDE_EXCEEDED)
}

/// Add the coverage-boundary term from the Tile_Compositor seam objective.
/// Overlap disagreement and the boundary penalty share one cost unit, so a
/// candidate inside the 16-pixel boundary band always pays at least 1.0.
///
/// The function is intentionally pure so the production seam adapter and the
/// exhaustive property oracle use exactly the same cost contract.
pub(crate) fn seam_candidate_cost(overlap_disagreement: f64, boundary_distance: f64) -> f64 {
    overlap_disagreement.max(0.0)
        + f64::from((boundary_distance.is_finite() && boundary_distance < 16.0) as u8)
}

/// Add current-overlap edge alignment to the station seam objective. The
/// alignment value is content-derived and non-negative; an aligned edge keeps
/// the existing cost exactly, while a seam through a missing or differently
/// oriented stroke is more expensive even when its colour disagreement is
/// deceptively small.
pub(crate) fn seam_candidate_cost_with_alignment(
    overlap_disagreement: f64,
    boundary_distance: f64,
    edge_misalignment: f64,
) -> f64 {
    seam_candidate_cost(overlap_disagreement, boundary_distance) + edge_misalignment.max(0.0)
}

/// Minimum binary ownership cut for one incoming station. Coverage seeds are
/// hard constraints; the weak interior prior only breaks equally good paths.
/// The seam cost penalizes the native 16px coverage boundary band. Covered
/// pixels are not penalized merely for belonging to that source: such an
/// area penalty can force a short, visibly worse cut along its coverage edge.
pub(crate) fn layered_seam_cut(
    width: usize,
    height: usize,
    preference: &[f64],
    disagreement: &[f64],
    base_distance: &[f64],
    candidate_distance: &[f64],
    fixed: &[i8],
) -> Vec<u8> {
    layered_seam_cut_with_existing_boundaries(
        width,
        height,
        preference,
        disagreement,
        base_distance,
        candidate_distance,
        fixed,
        &vec![(0.0, 0.0); width * height],
    )
}

pub(crate) fn existing_owner_boundary_costs(
    tiles: &[super::tone::ToneTile],
    owners: &[u16],
    distances: &[f64],
    width: usize,
    height: usize,
    bounds: (u32, u32, u32, u32),
    step: u32,
) -> Vec<(f64, f64)> {
    let count = width * height;
    assert_eq!(owners.len(), count);
    assert_eq!(distances.len(), count);
    let (left, top, right, bottom) = bounds;
    let mut pairs = std::collections::BTreeMap::new();
    let mut costs = vec![(0.0, 0.0); count];
    let position = |i: usize| {
        (
            (left + (i % width) as u32 * step + step / 2).min(right),
            (top + (i / width) as u32 * step + step / 2).min(bottom),
        )
    };
    for i in 0..count {
        for (other, horizontal) in [
            ((i % width + 1 < width).then_some(i + 1), true),
            ((i / width + 1 < height).then_some(i + width), false),
        ] {
            let Some(other) = other else {
                continue;
            };
            let (a, b) = (owners[i], owners[other]);
            if a == 0 || b == 0 || a == b {
                continue;
            }
            let Some(first) = tiles.iter().position(|tile| tile.owner_id == a) else {
                continue;
            };
            let Some(second) = tiles.iter().position(|tile| tile.owner_id == b) else {
                continue;
            };
            let key = (first.min(second), first.max(second));
            let (p, q) = (position(i), position(other));
            let disagreement = pairs
                .entry(key)
                .or_insert_with(|| super::tone::PairField::new(&tiles[key.0], &tiles[key.1]))
                .as_ref()
                .and_then(|field| {
                    field.seam_disagreement(
                        (f64::from(p.0) + f64::from(q.0)) * 0.5,
                        (f64::from(p.1) + f64::from(q.1)) * 0.5,
                    )
                })
                .unwrap_or(0.0)
                * 0.1;
            let cost =
                0.005 + seam_candidate_cost(disagreement, distances[i].min(distances[other]));
            if horizontal {
                costs[i].0 = cost;
            } else {
                costs[i].1 = cost;
            }
        }
    }
    costs
}

/// One alpha move must charge seams between different existing owners too.
/// Otherwise preserving an old visible seam costs zero and a later source
/// cannot improve it. Project non-submodular edge costs conservatively to
/// the representable binary energy; hard coverage constraints remain fixed.
#[allow(clippy::too_many_arguments)]
pub(crate) fn layered_seam_cut_with_existing_boundaries(
    width: usize,
    height: usize,
    preference: &[f64],
    disagreement: &[f64],
    base_distance: &[f64],
    candidate_distance: &[f64],
    fixed: &[i8],
    existing_boundary_costs: &[(f64, f64)],
) -> Vec<u8> {
    let count = width * height;
    assert_eq!(existing_boundary_costs.len(), count);
    assert!(
        [
            preference.len(),
            disagreement.len(),
            base_distance.len(),
            candidate_distance.len(),
            fixed.len()
        ]
        .iter()
        .all(|&n| n == count)
    );
    let mut preference = preference
        .iter()
        .map(|value| value * 0.02)
        .collect::<Vec<_>>();
    let costs = (0..count)
        .map(|i| seam_candidate_cost(disagreement[i], base_distance[i].min(candidate_distance[i])))
        .collect::<Vec<_>>();
    let mut edges = (0..count)
        .map(|i| {
            let weight = |other: usize| 0.005 + (costs[i] + costs[other]) * 0.5;
            (
                (i % width + 1 < width).then(|| weight(i + 1)),
                (i / width + 1 < height).then(|| weight(i + width)),
            )
        })
        .collect::<Vec<_>>();
    for i in 0..count {
        for (other, old_cost, horizontal) in [
            (
                (i % width + 1 < width).then_some(i + 1),
                existing_boundary_costs[i].0,
                true,
            ),
            (
                (i / width + 1 < height).then_some(i + width),
                existing_boundary_costs[i].1,
                false,
            ),
        ] {
            let Some(other) = other else {
                continue;
            };
            let weight = if horizontal {
                edges[i].0.as_mut()
            } else {
                edges[i].1.as_mut()
            }
            .unwrap();
            let retained_cost = old_cost.max(0.0).min(2.0 * (*weight));
            *weight -= retained_cost * 0.5;
            preference[i] += retained_cost * 0.5;
            preference[other] += retained_cost * 0.5;
        }
    }
    super::super::seam_cut::cut_grid_weighted(
        width,
        height,
        &preference,
        disagreement,
        fixed,
        &edges,
    )
}

#[derive(Debug, Clone)]
pub(crate) struct StationOwnershipPlan {
    pub(crate) step: u32,
    pub(crate) width: usize,
    pub(crate) height: usize,
    pub(crate) owners: Vec<u16>,
}

impl StationOwnershipPlan {
    pub(crate) fn owner_at(&self, x: u32, y: u32) -> u16 {
        self.owners[((y / self.step) as usize).min(self.height - 1) * self.width
            + ((x / self.step) as usize).min(self.width - 1)]
    }
}

/// A thin highlight's amplitude is normalised by the source samples that
/// contain it, rather than by the dark 64px background underneath it. Otherwise
/// a valid native contour can overwhelm even a large unsupported colour step.
fn native_detail_difference(a: [f32; 3], b: [f32; 3], da: [f32; 3], db: [f32; 3]) -> f64 {
    (0..3)
        .map(|c| {
            let level = ((a[c] + da[c] + b[c] + db[c]) * 0.5).max(0.02);
            // A difference between nonnegative source intensities divided by their
            // midpoint is at most two. Preserve that bound when the interpolated
            // low field changes slightly across the tangent probe.
            f64::from((da[c] - db[c]) / level).powi(2).min(4.0)
        })
        .sum::<f64>()
        / 3.0
}

#[cfg(test)]
#[test]
fn narrow_highlight_cost_uses_its_supported_source_level() {
    let cost = native_detail_difference([0.02; 3], [0.02; 3], [0.9; 3], [0.0; 3]);
    assert!(cost > 3.0 && cost <= 4.0);
    assert_eq!(
        native_detail_difference([0.02; 3], [0.02; 3], [0.9; 3], [0.9; 3]),
        0.0
    );
}

/// Revisit every station after all native fields exist, accepting only moves
/// that lower the actual multi-owner energy. Sequential insertion cannot
/// revisit a poor early seam with a station that has already been rendered.
pub(crate) fn refine_station_ownership(
    tiles: &[super::tone::ToneTile],
    relations: &[(usize, usize)],
    owners: &[u16],
    output_width: u32,
    output_height: u32,
) -> Option<StationOwnershipPlan> {
    // This is a read-only cost predictor for ownership planning, before
    // final RAW reconstruction. The final harmonizer still receives an
    // immutable ownership plane and fits the ultimately chosen boundaries.
    let calibration = tiles
        .iter()
        .any(|tile| tile.native_detail.is_some())
        .then(|| {
            super::tone_local::fit_local_tone(tiles, relations, owners, output_width, output_height)
        })
        .flatten();
    refine_station_ownership_with_calibration(
        tiles,
        relations,
        owners,
        output_width,
        output_height,
        calibration.as_ref(),
    )
}

fn refine_station_ownership_with_calibration(
    tiles: &[super::tone::ToneTile],
    relations: &[(usize, usize)],
    owners: &[u16],
    output_width: u32,
    output_height: u32,
    calibration: Option<&super::tone_local::LocalToneField>,
) -> Option<StationOwnershipPlan> {
    use super::tone;
    if tiles.len() < 2
        || output_width == 0
        || output_height == 0
        || owners.len() != output_width as usize * output_height as usize
    {
        return None;
    }
    // Match the native 16px analysis cells rather than grouping three of
    // them into one ownership label. Otherwise a thin edge can zigzag
    // between sources inside a coarse cell despite the boundary probes.
    let mut step = 16u32;
    while u64::from(output_width.div_ceil(step))
        * u64::from(output_height.div_ceil(step))
        * tiles.len() as u64
        > 40_000_000
    {
        if step >= 64 {
            return None;
        }
        step *= 2;
    }
    let width = output_width.div_ceil(step) as usize;
    let height = output_height.div_ceil(step) as usize;
    let count = width * height;
    let position = |i: usize| {
        (
            (i % width) as f64 * step as f64 + (step as f64 - 1.0) * 0.5,
            (i / width) as f64 * step as f64 + (step as f64 - 1.0) * 0.5,
        )
    };
    let mut labels = (0..count)
        .map(|i| {
            let (x, y) = position(i);
            owners[(y.round() as u32).min(output_height - 1) as usize * output_width as usize
                + (x.round() as u32).min(output_width - 1) as usize]
        })
        .collect::<Vec<_>>();
    let mut colors = Vec::new();
    let mut details = Vec::new();
    let mut valid = Vec::new();
    let mut near_boundary = Vec::new();
    for tile in tiles {
        let mut rgb = Vec::with_capacity(count);
        let mut detail = Vec::with_capacity(count);
        let mut covered = Vec::with_capacity(count);
        let mut boundary = Vec::with_capacity(count);
        for cell in 0..count {
            let (x, y) = position(cell);
            rgb.push(tone::sample_tile_bilinear(tile, x, y).unwrap_or([0.0; 3]));
            // Probe the actual pixel-label boundary, not the cell centre.
            // Three tangent probes preserve narrow edges that sigma=64 hides.
            detail.push(std::array::from_fn::<_, 2, _>(|axis| {
                std::array::from_fn::<_, 3, _>(|sample| {
                    let tangent = (sample as f64 - 1.0) * step as f64 / 3.0;
                    let (px, py) = if axis == 0 {
                        (x + step as f64 * 0.5, y + tangent)
                    } else {
                        (x + tangent, y + step as f64 * 0.5)
                    };
                    tone::sample_tile_detail_bilinear(tile, px, py).unwrap_or([0.0; 3])
                })
            }));
            let u = (x - tile.world_origin.0 - (tile.world_stride - 1.0) * 0.5) / tile.world_stride;
            let v = (y - tile.world_origin.1 - (tile.world_stride - 1.0) * 0.5) / tile.world_stride;
            let (cx, cy) = (u.floor() as i64, v.floor() as i64);
            // A conservative one-cell (16 native pixels) coverage band.
            // Partially covered quadrature cells count as a boundary too.
            boundary.push((-1..=2).any(|dy| {
                (-1..=2).any(|dx| {
                    let (px, py) = (cx + dx, cy + dy);
                    let invalid = px < 0
                        || py < 0
                        || px >= tile.validity.width() as i64
                        || py >= tile.validity.height() as i64
                        || tile.validity.get_pixel(px as u32, py as u32)[0]
                            != tone::TONE_CELL_SAMPLE;
                    let left = tile.world_origin.0 + px as f64 * tile.world_stride;
                    let top = tile.world_origin.1 + py as f64 * tile.world_stride;
                    let dx = (left - x).max(x - (left + tile.world_stride)).max(0.0);
                    let dy = (top - y).max(y - (top + tile.world_stride)).max(0.0);
                    invalid && dx.hypot(dy) < 16.0
                })
            }));
            covered.push(
                u >= 0.0
                    && v >= 0.0
                    && (0..=1).all(|dy| {
                        (0..=1).all(|dx| {
                            let (px, py) = (cx + dx, cy + dy);
                            px >= 0
                                && py >= 0
                                && px < tile.validity.width() as i64
                                && py < tile.validity.height() as i64
                                && tile.validity.get_pixel(px as u32, py as u32)[0]
                                    == tone::TONE_CELL_SAMPLE
                        })
                    }),
            );
        }
        colors.push(rgb);
        details.push(detail);
        valid.push(covered);
        near_boundary.push(boundary);
    }
    let maximum = tiles.iter().map(|t| t.owner_id).max()? as usize;
    let mut index = vec![None; maximum + 1];
    for (i, tile) in tiles.iter().enumerate() {
        index[tile.owner_id as usize] = Some(i)
    }
    let calibrated_pairs = calibration.map(|_| {
        let mut fields = std::collections::HashMap::new();
        for &(a, b) in relations {
            if a < tiles.len() && b < tiles.len() {
                let pair = (a.min(b), a.max(b));
                fields
                    .entry(pair)
                    .or_insert_with(|| tone::PairField::new(&tiles[pair.0], &tiles[pair.1]));
            }
        }
        fields
    });
    let cost = |a: u16, b: u16, cell: usize, horizontal: bool| {
        if a == b || a == 0 || b == 0 {
            return 0.0;
        }
        let (Some(first), Some(second)) = (
            index.get(a as usize).copied().flatten(),
            index.get(b as usize).copied().flatten(),
        ) else {
            return 100.0;
        };
        let calibration_cost =
            if let (Some(model), Some(fields)) = (calibration, calibrated_pairs.as_ref()) {
                let pair = (first.min(second), first.max(second));
                let (x, y) = position(cell);
                let Some((a, b)) = fields
                    .get(&pair)
                    .and_then(|field| field.as_ref())
                    .and_then(|field| field.at(x, y))
                else {
                    return 10_000.0;
                };
                let delta = tone::delta_e00_rgb(
                    model.corrected(pair.0, a, x, y),
                    model.corrected(pair.1, b, x, y),
                );
                100.0
                    * (delta - super::report::TONE_BOUNDARY_DELTA_E_THRESHOLD)
                        .max(0.0)
                        .powi(2)
            } else {
                0.0
            };
        let (a, b) = (colors[first][cell], colors[second][cell]);
        // A photometrically unvalidated pair cannot supply a calibrated seam.
        // Keep its actual RGB available for diagnostics, but route ownership
        // through validated overlaps wherever coverage permits that route.
        if !relations
            .iter()
            .any(|&(a, b)| (a == first && b == second) || (a == second && b == first))
        {
            return 10_000.0;
        }
        let midpoint: [f32; 3] = std::array::from_fn(|c| (a[c] + b[c]) * 0.5);
        let bounded = |source: [f32; 3]| {
            std::array::from_fn(|c| {
                midpoint[c].clamp(
                    tone::TONE_MIN_GAIN * source[c] - tone::TONE_MAX_ABS_OFFSET,
                    tone::TONE_MAX_GAIN * source[c] + tone::TONE_MAX_ABS_OFFSET,
                )
            })
        };
        let gap = tone::delta_e00_rgb(bounded(a), bounded(b));
        let raw = tone::delta_e00_rgb(a, b);
        let fully_calibratable = [a, b].iter().all(|rgb| {
            rgb.iter().all(|value| {
                (tone::TONE_MIN_SAMPLE_LUMINANCE..=tone::TONE_MAX_SAMPLE_LUMINANCE).contains(value)
            })
        });
        // Broad illumination can be harmonised, but a displaced stroke or
        // frame edge cannot. Charge disagreement of the immutable native
        // cell detail, without changing either source or its geometry.
        let axis = usize::from(!horizontal);
        let detail_disagreement = (0..3)
            .map(|sample| {
                if tiles[first].native_detail.is_some() && tiles[second].native_detail.is_some() {
                    return native_detail_difference(
                        a,
                        b,
                        details[first][cell][axis][sample],
                        details[second][cell][axis][sample],
                    );
                }
                (0..3)
                    .map(|c| {
                        let scale = ((a[c] + b[c]) * 0.5).max(0.02);
                        f64::from(
                            (details[first][cell][axis][sample][c]
                                - details[second][cell][axis][sample][c])
                                / scale,
                        )
                        .powi(2)
                    })
                    .sum::<f64>()
                    / 3.0
            })
            .fold(0.0f64, f64::max);
        f64::from(near_boundary[first][cell] || near_boundary[second][cell])
            + 0.005
            + 100.0 * detail_disagreement
            + calibration_cost

            // Clipped channels cannot supply the ordinary-range affine fit.
            // Route a chromatic highlight step away from that unsupported
            // calibration rather than extrapolating it into a visible seam.
            + if fully_calibratable {
                0.0
            } else {
                100.0
                    * (raw - super::report::TONE_BOUNDARY_DELTA_E_THRESHOLD)
                        .max(0.0)
                        .powi(2)
            }
            + 0.1
                * (raw / super::report::TONE_BOUNDARY_DELTA_E_THRESHOLD)
                    .powi(2)
                    .min(64.0)
            + 100.0
                * (gap - super::report::TONE_BOUNDARY_DELTA_E_THRESHOLD)
                    .max(0.0)
                    .powi(2)
    };
    let energy = |labels: &[u16]| {
        (0..count)
            .map(|i| {
                (if i % width + 1 < width {
                    cost(labels[i], labels[i + 1], i, true)
                } else {
                    0.0
                }) + (if i / width + 1 < height {
                    cost(labels[i], labels[i + width], i, false)
                } else {
                    0.0
                })
            })
            .sum::<f64>()
    };
    for _ in 0..3 {
        let mut changed = false;
        for (candidate, tile) in tiles.iter().enumerate() {
            let alpha = tile.owner_id;
            let fixed = (0..count)
                .map(|i| {
                    if labels[i] == alpha {
                        1
                    } else if labels[i] == 0 || !valid[candidate][i] {
                        -1
                    } else {
                        0
                    }
                })
                .collect::<Vec<_>>();
            let mut preference = vec![0.0; count];
            let mut edges = vec![(None, None); count];
            for i in 0..count {
                for (other, horizontal) in [
                    ((i % width + 1 < width).then_some(i + 1), true),
                    ((i / width + 1 < height).then_some(i + width), false),
                ] {
                    let Some(other) = other else { continue };
                    let (e00, mut e01, mut e10) = (
                        cost(labels[i], labels[other], i, horizontal),
                        cost(labels[i], alpha, i, horizontal),
                        cost(alpha, labels[other], i, horizontal),
                    );
                    // Majorize the mixed states instead of lowering E00.
                    // Lowering E00 erases the actual cost of an unsupported
                    // old seam, so a covered bridge can never improve it in
                    // the binary surrogate. This upper bound is exact at the
                    // current labels and the all-alpha state.
                    let deficit = (e00 - e01 - e10).max(0.0) * 0.5;
                    e01 += deficit;
                    e10 += deficit;
                    let weight = (e01 + e10 - e00).max(0.0) * 0.5;
                    preference[i] += e00 - e10 + weight;
                    preference[other] += e00 - e01 + weight;
                    if horizontal {
                        edges[i].0 = Some(weight)
                    } else {
                        edges[i].1 = Some(weight)
                    }
                }
            }
            let cut = super::super::seam_cut::cut_grid_weighted_compact_fixed(
                width,
                height,
                &preference,
                &vec![0.0; count],
                &fixed,
                &edges,
            );
            let proposed = labels
                .iter()
                .zip(cut)
                .map(|(&old, new)| if new > 0 { alpha } else { old })
                .collect::<Vec<_>>();
            if energy(&proposed) + 1e-6 < energy(&labels) {
                labels = proposed;
                changed = true
            }
        }
        if !changed {
            break;
        }
    }
    Some(StationOwnershipPlan {
        step,
        width,
        height,
        owners: labels,
    })
}

/// Find the deterministic minimum-cost vertical seam in a row-major overlap
/// grid.  A path contains one column per row and may move at most one column
/// between adjacent rows.  This pure solver is shared by the production seam
/// adapter and Property 53's exhaustive oracle.
pub(crate) fn minimum_vertical_seam(
    costs: &[f64],
    width: usize,
    height: usize,
) -> Option<(Vec<usize>, f64)> {
    if width == 0 || height == 0 || costs.len() != width.saturating_mul(height) {
        return None;
    }
    let mut previous = costs[..width].to_vec();
    let mut predecessors = vec![0usize; width.saturating_mul(height)];
    for row in 1..height {
        let mut current = vec![f64::INFINITY; width];
        for column in 0..width {
            let start = column.saturating_sub(1);
            let end = (column + 1).min(width - 1);
            let (best_column, best_cost) = (start..=end)
                .map(|candidate| (candidate, previous[candidate]))
                .min_by(|(left_column, left_cost), (right_column, right_cost)| {
                    left_cost
                        .total_cmp(right_cost)
                        .then_with(|| left_column.cmp(right_column))
                })?;
            current[column] = best_cost + costs[row * width + column];
            predecessors[row * width + column] = best_column;
        }
        previous = current;
    }
    let (mut column, total) = previous.iter().copied().enumerate().min_by(
        |(left_column, left_cost), (right_column, right_cost)| {
            left_cost
                .total_cmp(right_cost)
                .then_with(|| left_column.cmp(right_column))
        },
    )?;
    let mut path = vec![0usize; height];
    path[height - 1] = column;
    for row in (1..height).rev() {
        column = predecessors[row * width + column];
        path[row - 1] = column;
    }
    Some((path, total))
}

/// For an overlap narrower than 32 world pixels, the requirement-prescribed
/// seam is its geometric centre line.  The boolean identifies a vertical
/// line; the returned coordinates are one cross-axis coordinate per along-axis
/// row/column and are deterministic for even widths/heights.
pub(crate) fn narrow_overlap_centerline(width: usize, height: usize) -> Option<(bool, Vec<usize>)> {
    if width == 0 || height == 0 || width.max(height) >= 32 {
        return None;
    }
    if width <= height {
        Some((true, vec![width / 2; height]))
    } else {
        Some((false, vec![height / 2; width]))
    }
}

/// Decide and record the actual pipeline path from the Station_Grouper output
/// and the persisted compositor setting (requirements 15.1, 15.3 and 15.11).
///
/// `Some(0)` and `Some(1)` are deliberately the same station-layer-only
/// boundary: neither may enter inter-station pose solving or a tile compositor.
/// `None` means grouping never produced an outcome, so the existing legacy
/// fallback is the only truthful path identifier. The explicit legacy
/// comparison setting always wins, including for a single-station run.
pub(crate) fn select_and_record_run_path(
    station_count: Option<usize>,
    compositor: StackCompositorChoice,
    recorder: Option<&StackReportRecorder>,
) -> StackPathSelection {
    let selection = if compositor == StackCompositorChoice::LegacySingleLayerMosaic {
        StackPathSelection {
            selected_path: SelectedPath::LegacySingleLayerMosaic,
            use_virtual_tiles: false,
        }
    } else {
        match station_count {
            Some(0 | 1) => StackPathSelection {
                selected_path: SelectedPath::SingleStation,
                use_virtual_tiles: false,
            },
            Some(_) => StackPathSelection {
                selected_path: compositor.selected_path(),
                use_virtual_tiles: true,
            },
            None => StackPathSelection {
                selected_path: SelectedPath::LegacySingleLayerMosaic,
                use_virtual_tiles: false,
            },
        }
    };
    if let Some(recorder) = recorder {
        recorder.set_selected_path(selection.selected_path);
    }
    selection
}

#[cfg(test)]
mod tests;
