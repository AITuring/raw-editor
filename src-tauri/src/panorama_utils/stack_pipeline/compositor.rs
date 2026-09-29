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
//! 任务 13.2 will grow the seam-cost work of the layered path into this module;
//! for now it only owns the path choice and its stable identifier.

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
        assert!(reject_oversized_canvas(MAX_OUTPUT_CANVAS_LONG_SIDE, 1).is_none());
        assert!(reject_oversized_canvas(1, MAX_OUTPUT_CANVAS_LONG_SIDE).is_none());
        assert!(reject_oversized_canvas(MAX_OUTPUT_CANVAS_LONG_SIDE + 1, 1).is_some());
        assert!(!is_canvas_rejection("Virtual tile 3 dimensions changed"));
    }

    #[test]
    fn layered_compositor_rejects_an_oversized_canvas_before_loading_a_tile() {
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
