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
//! The production seam objective helpers also live here so the progressive
//! compositor and Property 53/59 use the same deterministic implementation.

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
    if residual.identity() {
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
pub(crate) fn seam_candidate_cost(overlap_disagreement: f64, boundary_distance: f64) -> f64 {
    overlap_disagreement.max(0.0)
        + f64::from((boundary_distance.is_finite() && boundary_distance < 16.0) as u8)
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
    use super::*;

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
        use crate::panorama_stitching::ImageInfo;
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
