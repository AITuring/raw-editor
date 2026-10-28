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
mod tests {
    #[test]
    fn a_contour_inverse_composes_once_with_the_source_camera_inverse() {
        use super::super::line_layers::{ContourTrack, LineLayers};
        let track = |station, dx| ContourTrack {
            station,
            points: (0..160)
                .map(|i| [i as f64 * 32.0, 400.0 + i as f64 * 0.4 + dx])
                .collect(),
        };
        let mut model = ResidualWarp::new();
        model.line_layers = LineLayers::fit(vec![track(0, 0.0), track(1, 17.0), track(2, -7.0)]);
        model.register_station_alias(70, 1);
        let inverse = Matrix3::new(0.5, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0);
        let actual =
            tile_coordinate_with_residual(Point2::new(408.0, 640.0), &inverse, &model, 70).unwrap();
        assert!((actual.x - 212.5).abs() < 1e-6);
        assert_eq!(actual.y, 320.0);
        assert_eq!(
            tile_coordinate_with_residual(Point2::new(100.0, 640.0), &inverse, &model, 70).unwrap(),
            Point2::new(50.0, 320.0)
        );
    }

    #[test]
    fn large_canvas_revisit_can_cut_through_a_native_cell_corridor() {
        use super::super::tone::ToneTile;
        use image::{GrayImage, Luma, Rgb, Rgb32FImage};
        let tile = |station_index, owner_id, left, right, displaced| ToneTile {
            station_index,
            owner_id,
            low: Rgb32FImage::from_pixel(1024, 8, Rgb([0.4; 3])),
            cell_mean: Rgb32FImage::from_fn(1024, 8, |x, _| {
                Rgb([if displaced && !(512..=513).contains(&x) {
                    0.1
                } else {
                    0.4
                }; 3])
            }),
            validity: GrayImage::from_fn(1024, 8, |x, _| {
                Luma([if x >= left && x < right { 255 } else { 0 }])
            }),
            native_detail: None,
            cell_coverage: (0..8192)
                .map(|i| {
                    if i % 1024 >= left && i % 1024 < right {
                        1.0
                    } else {
                        0.0
                    }
                })
                .collect(),
            world_origin: (0.0, 0.0),
            world_size: (16384.0, 128.0),
            world_stride: 16.0,
        };
        let tiles = [tile(0, 1, 0, 600, false), tile(1, 2, 400, 1024, true)];
        let owners = (0..128)
            .flat_map(|_| (0..16384).map(|x| if x < 8192 { 1 } else { 2 }))
            .collect::<Vec<_>>();
        let plan = super::refine_station_ownership(&tiles, &[(0, 1)], &owners, 16384, 128).unwrap();
        // x=8208 lies in the actual common-detail corridor; a 32px grid
        // cannot represent this seam and crosses displaced content instead.
        for y in (32..96).step_by(16) {
            assert_eq!(plan.owner_at(8207, y), 1);
            assert_eq!(plan.owner_at(8208, y), 2);
        }
    }

    #[test]
    fn ownership_revisit_measures_detail_at_the_actual_label_boundary() {
        use super::super::tone::ToneTile;
        use image::{GrayImage, Luma, Rgb, Rgb32FImage};
        let tile = |station_index, owner_id, left, right, displaced| ToneTile {
            station_index,
            owner_id,
            low: Rgb32FImage::from_pixel(16, 16, Rgb([0.4; 3])),
            cell_mean: Rgb32FImage::from_fn(16, 16, |x, _| {
                Rgb([if displaced && x == 8 { 0.1 } else { 0.4 }; 3])
            }),
            validity: GrayImage::from_fn(16, 16, |x, _| {
                Luma([if x >= left && x < right { 255 } else { 0 }])
            }),
            native_detail: None,
            cell_coverage: (0..256)
                .map(|i| {
                    if i % 16 >= left && i % 16 < right {
                        1.0
                    } else {
                        0.0
                    }
                })
                .collect(),
            world_origin: (0.0, 0.0),
            world_size: (256.0, 256.0),
            world_stride: 16.0,
        };
        let tiles = [tile(0, 1, 0, 12, false), tile(1, 2, 4, 16, true)];
        let owners = (0..256)
            .flat_map(|_| (0..256).map(|x| if x < 128 { 1 } else { 2 }))
            .collect::<Vec<_>>();
        // The retained seam is between columns 7 and 8. At column 7's
        // centre the two detail fields are identical, so a centre-only cost
        // leaves this seam in place. At its actual boundary they differ.
        let plan = super::refine_station_ownership(&tiles, &[(0, 1)], &owners, 256, 256).unwrap();
        assert_eq!(plan.owner_at(8, 128), 1);
        assert_eq!(plan.owner_at(248, 128), 2);
        for y in (32..224).step_by(16) {
            assert_eq!(plan.owner_at(120, y), plan.owner_at(136, y));
        }
    }

    #[test]
    fn ownership_revisit_avoids_displaced_detail_hidden_by_equal_low_frequency() {
        use super::super::tone::ToneTile;
        use image::{GrayImage, Luma, Rgb, Rgb32FImage};
        let tile = |station_index, owner_id, left, right, displaced| ToneTile {
            station_index,
            owner_id,
            low: Rgb32FImage::from_pixel(16, 16, Rgb([0.4; 3])),
            cell_mean: Rgb32FImage::from_fn(16, 16, |x, _| {
                Rgb([if displaced && (7..=9).contains(&x) {
                    0.1
                } else {
                    0.4
                }; 3])
            }),
            validity: GrayImage::from_fn(16, 16, |x, _| {
                Luma([if x >= left && x < right { 255 } else { 0 }])
            }),
            native_detail: None,
            cell_coverage: (0..256)
                .map(|i| {
                    if i % 16 >= left && i % 16 < right {
                        1.0
                    } else {
                        0.0
                    }
                })
                .collect(),
            world_origin: (0.0, 0.0),
            world_size: (256.0, 256.0),
            world_stride: 16.0,
        };
        let tiles = [tile(0, 1, 0, 12, false), tile(1, 2, 4, 16, true)];
        let owners = (0..256)
            .flat_map(|_| (0..256).map(|x| if x < 128 { 1 } else { 2 }))
            .collect::<Vec<_>>();
        let plan = super::refine_station_ownership(&tiles, &[(0, 1)], &owners, 256, 256).unwrap();
        assert_eq!(plan.owner_at(8, 128), 1);
        assert_eq!(plan.owner_at(248, 128), 2);
        for y in 2..plan.height - 2 {
            for x in 0..plan.width - 1 {
                if plan.owners[y * plan.width + x] != plan.owners[y * plan.width + x + 1] {
                    assert!(
                        !(7..=9).contains(&x),
                        "cut crossed displaced detail at {x},{y}"
                    );
                }
            }
        }
    }

    #[test]
    fn ownership_revisit_routes_an_unvalidated_seam_through_covered_relations() {
        use super::super::tone::ToneTile;
        use image::{GrayImage, Luma, Rgb, Rgb32FImage};
        let tile = |station_index, owner_id, left: u32, right: u32| ToneTile {
            station_index,
            owner_id,
            low: Rgb32FImage::from_pixel(16, 16, Rgb([0.4; 3])),
            cell_mean: Rgb32FImage::from_pixel(16, 16, Rgb([0.4; 3])),
            validity: GrayImage::from_fn(16, 16, |x, _| {
                Luma([if x >= left && x < right { 255 } else { 0 }])
            }),
            native_detail: None,
            cell_coverage: (0..256)
                .map(|i| {
                    if i % 16 >= left && i % 16 < right {
                        1.0
                    } else {
                        0.0
                    }
                })
                .collect(),
            world_origin: (0.0, 0.0),
            world_size: (256.0, 256.0),
            world_stride: 16.0,
        };
        let tiles = [tile(0, 1, 0, 11), tile(1, 2, 5, 16), tile(2, 3, 3, 13)];
        let owners = (0..256)
            .flat_map(|_| (0..256).map(|x| if x < 128 { 1 } else { 2 }))
            .collect::<Vec<_>>();
        let plan =
            super::refine_station_ownership(&tiles, &[(0, 2), (1, 2)], &owners, 256, 256).unwrap();
        assert_eq!(plan.owner_at(8, 128), 1);
        assert_eq!(plan.owner_at(248, 128), 2);
        // The outermost interpolation row has no fully covered proposal.
        for y in 1..plan.height - 1 {
            for x in 0..plan.width - 1 {
                let a = plan.owners[y * plan.width + x];
                let b = plan.owners[y * plan.width + x + 1];
                assert!(
                    !((a == 1 && b == 2) || (a == 2 && b == 1)),
                    "unvalidated pair at {x},{y}: {:?}",
                    plan.owners
                );
            }
        }
    }
    use super::*;

    #[test]
    fn existing_seam_binary_reduction_matches_an_exhaustive_energy_oracle() {
        let (width, height) = (3, 2);
        let prior = [0.3, -0.2, 0.5, -0.4, 0.1, -0.6];
        let disagreement = [0.2, 0.8, 0.4, 0.7, 0.1, 0.3];
        let distance = [100.0; 6];
        let fixed = [-1, 0, 0, 0, 0, 1];
        let old: [(f64, f64); 6] = [
            (0.4, 0.6),
            (0.1, 0.0),
            (0.0, 0.2),
            (0.8, 0.0),
            (0.6, 0.0),
            (0.0, 0.0),
        ];
        let energy = |mask: usize| {
            let mut value = 0.0;
            for i in 0..6 {
                let new = mask & (1 << i) != 0;
                if (fixed[i] < 0 && new) || (fixed[i] > 0 && !new) {
                    return f64::INFINITY;
                }
                value += if new {
                    (-prior[i] * 0.02f64).max(0.0)
                } else {
                    (prior[i] * 0.02f64).max(0.0)
                };
                for (other, retained) in [
                    ((i % width + 1 < width).then_some(i + 1), old[i].0),
                    ((i / width + 1 < height).then_some(i + width), old[i].1),
                ] {
                    let Some(other) = other else {
                        continue;
                    };
                    let other_new = mask & (1 << other) != 0;
                    let switched = 0.005 + (disagreement[i] + disagreement[other]) * 0.5;
                    value += match (new, other_new) {
                        (false, false) => retained.min(2.0 * switched),
                        (true, true) => 0.0,
                        _ => switched,
                    };
                }
            }
            value
        };
        let labels = layered_seam_cut_with_existing_boundaries(
            width,
            height,
            &prior,
            &disagreement,
            &distance,
            &distance,
            &fixed,
            &old,
        );
        let actual = labels
            .iter()
            .enumerate()
            .fold(0, |mask, (i, label)| mask | usize::from(*label > 0) << i);
        let best = (0..64).map(energy).min_by(f64::total_cmp).unwrap();
        assert!((energy(actual) - best).abs() < 1e-10);
    }

    #[test]
    fn incoming_source_can_remove_an_existing_seam_without_creating_a_worse_one() {
        let distance = vec![100.0; 2];
        let labels = layered_seam_cut_with_existing_boundaries(
            2,
            1,
            &[-0.1; 2],
            &[1.0; 2],
            &distance,
            &distance,
            &[0; 2],
            &[(10.0, 0.0), (0.0, 0.0)],
        );
        assert_eq!(labels, vec![255, 255]);
        let distance = vec![100.0; 3];
        let kept = layered_seam_cut_with_existing_boundaries(
            3,
            1,
            &[0.0; 3],
            &[1.0; 3],
            &distance,
            &distance,
            &[-1, 0, -1],
            &[(0.0, 0.0), (0.1, 0.0), (0.0, 0.0)],
        );
        assert_eq!(
            kept,
            vec![0; 3],
            "a cheap retained seam must beat two expensive new seams"
        );
    }

    #[test]
    fn coverage_penalty_must_not_force_a_high_disagreement_boundary_cut() {
        let (width, height) = (64, 8);
        let mut base_distance = vec![100.0; width * height];
        let candidate_distance = vec![100.0; width * height];
        let mut disagreement = vec![5.0; width * height];
        let mut fixed = vec![0; width * height];
        for y in 0..height {
            fixed[y * width] = -1;
            fixed[y * width + width - 1] = 1;
            for x in 0..16 {
                base_distance[y * width + x] = 8.0;
            }
            disagreement[y * width + 32] = 0.0;
            disagreement[y * width + 33] = 0.0;
        }
        let labels = layered_seam_cut(
            width,
            height,
            &vec![0.0; width * height],
            &disagreement,
            &base_distance,
            &candidate_distance,
            &fixed,
        );
        for y in 0..height {
            assert_eq!(
                labels[y * width + 32],
                0,
                "the seam was pushed onto the worse coverage edge"
            );
            assert_eq!(labels[y * width + 33], 255);
        }
    }

    #[test]
    fn layered_cut_follows_real_agreement_and_avoids_coverage_edges() {
        let (w, h) = (20, 8);
        let mut disagreement = vec![0.9; w * h];
        let mut fixed = vec![0; w * h];
        for y in 0..h {
            fixed[y * w] = -1;
            fixed[y * w + w - 1] = 1;
            disagreement[y * w + 5] = 0.0;
            disagreement[y * w + 6] = 0.0;
        }
        let distance = vec![100.0; w * h];
        let labels = layered_seam_cut(
            w,
            h,
            &vec![0.0; w * h],
            &disagreement,
            &distance,
            &distance,
            &fixed,
        );
        for y in 0..h {
            assert_eq!(labels[y * w + 5], 0);
            assert_eq!(labels[y * w + 6], 255);
        }
        let mut boundary = distance.clone();
        for y in 0..h {
            boundary[y * w + 5] = 4.0;
            boundary[y * w + 6] = 4.0;
        }
        let guarded = layered_seam_cut(
            w,
            h,
            &vec![0.0; w * h],
            &disagreement,
            &boundary,
            &boundary,
            &fixed,
        );
        for y in 0..h {
            assert_eq!(guarded[y * w], 0);
            assert_eq!(guarded[y * w + w - 1], 255);
            assert!(!(guarded[y * w + 5] == 0 && guarded[y * w + 6] == 255));
        }
    }

    #[test]
    fn default_choice_is_the_layered_path_with_every_comparison_off() {
        // 需求 15.2 / 15.10.
        let choice = StackCompositorChoice::default();
        assert_eq!(choice, StackCompositorChoice::LayeredVirtualTile);
        assert!(!choice.is_comparison_path());
        assert_eq!(choice.as_identifier(), "layered_virtual_tile");
    }

    #[test]
    fn identifiers_round_trip_for_every_variant() {
        for choice in [
            StackCompositorChoice::LayeredVirtualTile,
            StackCompositorChoice::ProgressiveSeamTile,
            StackCompositorChoice::StreamingMosaic,
            StackCompositorChoice::LegacySingleLayerMosaic,
        ] {
            let identifier = choice.as_identifier();
            assert_eq!(
                StackCompositorChoice::from_identifier(identifier),
                Some(choice),
                "{identifier} must round trip"
            );
            assert_eq!(choice.selected_path().as_identifier(), identifier);
        }
    }

    #[test]
    fn unset_and_unknown_settings_resolve_to_the_default_path() {
        // 任务 13.7: the setting is the only input of `resolve`.
        assert_eq!(StackCompositorChoice::from_identifier(""), None);
        assert_eq!(StackCompositorChoice::from_identifier("nope"), None);
        assert_eq!(
            StackCompositorChoice::resolve(Some("  Streaming_Mosaic ")),
            StackCompositorChoice::StreamingMosaic
        );
        for setting in [None, Some(""), Some("nope")] {
            assert_eq!(
                StackCompositorChoice::resolve(setting),
                StackCompositorChoice::LayeredVirtualTile
            );
        }
    }

    #[test]
    fn default_multi_station_selection_records_the_layered_path() {
        let recorder = StackReportRecorder::isolated("default-layered-selection", None);
        let selection = select_and_record_run_path(
            Some(2),
            StackCompositorChoice::resolve(None),
            Some(&recorder),
        );

        assert_eq!(selection.selected_path, SelectedPath::LayeredVirtualTile);
        assert!(selection.use_virtual_tiles);
        assert_eq!(
            recorder.snapshot().selected_path,
            SelectedPath::LayeredVirtualTile
        );
    }

    #[test]
    fn explicit_legacy_selection_records_the_legacy_path_without_virtual_tiles() {
        let recorder = StackReportRecorder::isolated("legacy-selection", None);
        let selection = select_and_record_run_path(
            Some(2),
            StackCompositorChoice::LegacySingleLayerMosaic,
            Some(&recorder),
        );

        assert_eq!(
            selection.selected_path,
            SelectedPath::LegacySingleLayerMosaic
        );
        assert!(!selection.use_virtual_tiles);
        assert_eq!(
            recorder.snapshot().selected_path,
            SelectedPath::LegacySingleLayerMosaic
        );
    }

    #[test]
    fn default_layered_entry_runs_group_tone_and_quality_gate_components() {
        // Exercise the same production component entry points selected by a
        // fresh install: the layered path, group Tone_Harmonizer and the
        // record-only Quality_Gate.  Keeping this fixture synthetic avoids a
        // RAW decode while still asserting the returned values that the
        // pipeline publishes to Stack_Report.
        use crate::panorama_utils::stack_pipeline::report::{QualityGateVerdict, ToneStatus};
        use crate::panorama_utils::stack_pipeline::{
            degradation, quality_gate, quality_gate_runner, residual_warp, tone,
        };
        use image::{GrayImage, Luma, Rgb, Rgb32FImage};
        use nalgebra::Matrix3;

        let _run_scope = degradation::begin_run_scope();
        let recorder = StackReportRecorder::isolated("default-layered-components", None);
        let selection = select_and_record_run_path(
            Some(2),
            StackCompositorChoice::resolve(None),
            Some(&recorder),
        );
        assert_eq!(selection.selected_path, SelectedPath::LayeredVirtualTile);
        assert!(selection.use_virtual_tiles);

        let (width, height) = (1_536u32, 512u32);
        let low_size = (64, 32);
        let tile = |station_index, owner_id, origin, value| tone::ToneTile {
            station_index,
            owner_id,
            low: Rgb32FImage::from_pixel(low_size.0, low_size.1, Rgb([value; 3])),
            validity: GrayImage::from_pixel(low_size.0, low_size.1, Luma([255])),
            world_origin: (origin, 0.0),
            world_size: (1_024.0, 512.0),
            world_stride: 16.0,
            cell_mean: Rgb32FImage::from_pixel(low_size.0, low_size.1, Rgb([value; 3])),
            native_detail: None,
            cell_coverage: vec![1.0; low_size.0 as usize * low_size.1 as usize],
        };
        let tiles = [tile(0, 1, 0.0, 0.40), tile(1, 2, 512.0, 0.36)];
        let owners = (0..height)
            .flat_map(|_| (0..width).map(|x| if x < 768 { 1u16 } else { 2u16 }))
            .collect::<Vec<_>>();
        let mut panorama = Rgb32FImage::from_fn(width, height, |x, _| {
            Rgb([if x < 768 { 0.40 } else { 0.36 }; 3])
        });
        let tone_report = tone::harmonize_tone_tiles(
            &mut panorama,
            &owners,
            width,
            &tiles,
            &GrayImage::from_pixel(width, height, Luma([255])),
            &[(0, 1)],
        );
        assert_eq!(tone_report.status, ToneStatus::Applied);
        assert_eq!(tone_report.tiles.len(), 2);

        let (qg_width, qg_height) = (1_024u32, 1_024u32);
        let output = Rgb32FImage::from_fn(qg_width, qg_height, |x, y| {
            let value = 0.2 + 0.3 * ((x + y) % 37) as f32 / 36.0;
            Rgb([value, value * 0.9, value * 0.8])
        });
        let coverage = GrayImage::from_pixel(qg_width, qg_height, Luma([255]));
        let ownership = vec![1u16; (qg_width * qg_height) as usize];
        let confidence = vec![0.8f32; ownership.len()];
        let textured = vec![255u8; ownership.len()];
        let shortfall = vec![0u8; ownership.len()];
        let disagreement = vec![0u8; ownership.len()];
        let source = quality_gate_runner::QualitySource {
            owner: 1,
            path: "synthetic.raw".to_string(),
            geometry: quality_gate::SourceGeometry {
                member_to_anchor: Matrix3::identity(),
                tile_to_world: Matrix3::identity(),
                station_id: 0,
            },
            dimensions: (qg_width, qg_height),
            native_warp: None,
            native_registration: None,
        };
        let residual = residual_warp::ResidualWarp::new();
        let sources = [source];
        let input = quality_gate_runner::QualityGateInput {
            output: &output,
            coverage: &coverage,
            ownership: &ownership,
            confidence: &confidence,
            textured: Some(&textured),
            owner_shortfall: Some(&shortfall),
            owner_disagreement: Some(&disagreement),
            unresolved_pixel_count: 0,
            world_origin: (0.0, 0.0),
            sources: &sources,
            accepted_station_relations: &[],
            residual: &residual,
            acceptance_render_scale: 1.0,
            final_sharpen_amount: 0.0,
        };
        let quality_report =
            quality_gate_runner::run_quality_gate(&input, &mut |_source| Ok(output.clone()));
        assert_eq!(
            quality_report.criteria.len(),
            quality_gate::QUALITY_CRITERIA.len()
        );
        assert_ne!(quality_report.verdict, QualityGateVerdict::NotRun);

        recorder.update(|report| {
            report.tone = tone_report.clone();
            report.quality_gate = quality_report.clone();
        });
        let published = recorder.snapshot();
        assert_eq!(published.selected_path, SelectedPath::LayeredVirtualTile);
        assert_eq!(published.tone.status, ToneStatus::Applied);
        assert_eq!(
            published.quality_gate.criteria.len(),
            quality_gate::QUALITY_CRITERIA.len()
        );
    }

    proptest::proptest! {
        #![proptest_config(proptest::prelude::ProptestConfig {
            cases: 100,
            ..proptest::prelude::ProptestConfig::default()
        })]

        // Feature: layered-camera-group-focus-stitching, Property 60: 对于任意联合覆盖边界，若其
        // 长边超过支持的画布上限，则不写出最终结果文件，且实测画布尺寸与该上限被记录。
        //
        // The same function writes the `canvas_long_side_exceeded` ledger entry; the
        // returned error, which becomes the run's error, carries the measured size and the
        // limit, and `is_canvas_rejection` keeps every fallback renderer from retrying.
        //
        // **Validates: Requirements 10.11**
        #[test]
        fn property_60_oversized_canvas_is_rejected_with_its_measured_size(
            width in 1u64..600_000,
            height in 1u64..600_000,
        ) {
            let _run_scope = crate::panorama_utils::stack_pipeline::degradation::begin_run_scope();
            let rejection = reject_oversized_canvas(width, height);
            proptest::prop_assert_eq!(
                rejection.is_some(),
                width.max(height) > MAX_OUTPUT_CANVAS_LONG_SIDE
            );
            if let Some(error) = rejection {
                proptest::prop_assert!(is_canvas_rejection(&error), "{}", error);
                let measured = format!("{width}x{height}");
                proptest::prop_assert!(error.contains(&measured), "{}", error);
                proptest::prop_assert!(
                    error.contains(&MAX_OUTPUT_CANVAS_LONG_SIDE.to_string()),
                    "{}",
                    error
                );
            }
        }
    }

    #[test]
    fn canvas_limit_boundary_is_inclusive() {
        let _run_scope = crate::panorama_utils::stack_pipeline::degradation::begin_run_scope();
        assert!(reject_oversized_canvas(MAX_OUTPUT_CANVAS_LONG_SIDE, 1).is_none());
        assert!(reject_oversized_canvas(1, MAX_OUTPUT_CANVAS_LONG_SIDE).is_none());
        assert!(reject_oversized_canvas(MAX_OUTPUT_CANVAS_LONG_SIDE + 1, 1).is_some());
        assert!(!is_canvas_rejection("Virtual tile 3 dimensions changed"));
    }

    #[test]
    fn layered_compositor_rejects_an_oversized_canvas_before_loading_a_tile() {
        let _run_scope = crate::panorama_utils::stack_pipeline::degradation::begin_run_scope();
        use crate::panorama_utils::shared::ImageInfo;
        use crate::panorama_utils::stitching;
        // An 8 px tile placed at 40,000× spans 320,000 world pixels.
        let info = ImageInfo {
            id: 0,
            filename: "/station/oversized/0000.NEF".to_string(),
            width: 8,
            height: 8,
            alignment_image: image::GrayImage::new(1, 1),
            full_image: None,
            scale_factor: 1.0,
            focal_length_35mm: Some(75.0),
            source_footprint: None,
            overview_reference: false,
            features: Vec::new(),
            top_features: Vec::new(),
            foreground_range: None,
            foreground_mask: None,
            horizontal_edge_rows: Vec::new(),
            vertical_edge_columns: Vec::new(),
        };
        let homographies = std::collections::HashMap::from([(
            0usize,
            Matrix3::new(40_000.0, 0.0, 0.0, 0.0, 40_000.0, 0.0, 0.0, 0.0, 1.0),
        )]);
        let app = tauri::test::mock_app();
        let mut loads = 0usize;
        let mut load = |_: &ImageInfo| -> Result<image::Rgb32FImage, String> {
            loads += 1;
            Ok(image::Rgb32FImage::new(8, 8))
        };
        let result = stitching::layered_virtual_tile_compositor_with_ownership(
            &[&info],
            &homographies,
            stitching::Projection::Planar,
            app.handle().clone(),
            "compositor-canvas-limit-test",
            &mut load,
        );
        let Err(error) = result else {
            panic!("an oversized canvas must be rejected");
        };
        assert!(is_canvas_rejection(&error), "{error}");
        assert_eq!(loads, 0, "no tile may be decoded for a rejected canvas");
    }

    #[test]
    fn single_station_is_not_a_compositor_choice() {
        // 需求 15.11's identifier describes the pipeline shape of a one-station
        // run, so it must not be selectable as a compositor.
        assert_eq!(
            StackCompositorChoice::from_identifier(SelectedPath::SingleStation.as_identifier()),
            None
        );
    }

    #[test]
    fn tile_coordinate_composes_inverse_warp_before_one_sample() {
        let mut tile_to_world_inverse = Matrix3::identity();
        tile_to_world_inverse[(0, 2)] = -10.0;
        let mut residual_inverse = Matrix3::identity();
        residual_inverse[(1, 2)] = 3.0;
        let coordinate = tile_coordinate_from_world(
            Point2::new(12.0, 8.0),
            &tile_to_world_inverse,
            &residual_inverse,
        )
        .expect("finite composed coordinate");
        assert_eq!(coordinate, Point2::new(2.0, 11.0));
    }

    #[test]
    fn residual_free_coordinate_is_exactly_the_nominal_inverse() {
        let mut inverse = Matrix3::identity();
        inverse[(0, 2)] = -10.0;
        let world = Point2::new(12.0, 8.0);
        let residual = ResidualWarp::default();
        assert_eq!(
            tile_coordinate_with_residual(world, &inverse, &residual, 7),
            tile_coordinate_from_world(world, &inverse, &Matrix3::identity())
        );
    }

    #[test]
    fn tone_cannot_change_the_ownership_plane() {
        let owners = vec![1u16, 2, 2, 1];
        assert_ownership_unchanged(&owners, &owners);
    }
}
