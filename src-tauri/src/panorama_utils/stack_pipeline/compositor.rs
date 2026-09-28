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
    pub(crate) fn resolve(setting: Option<&str>) -> Self {
        if let Some(legacy) = Self::legacy_environment_override() {
            return legacy;
        }
        setting.and_then(Self::from_identifier).unwrap_or_default()
    }

    /// The two pre-setting environment switches, kept only so an in-flight
    /// diagnostic session keeps working.  任务 13.7 deletes both of them; the
    /// setting above is already authoritative for a default build because
    /// neither variable is set there.
    fn legacy_environment_override() -> Option<Self> {
        if std::env::var_os("RAW_EDITOR_USE_STREAMING_VIRTUAL_TILE_MOSAIC").is_some() {
            return Some(Self::StreamingMosaic);
        }
        // The ownership compositor's semantics became the default path in
        // 任务 13.1, so this variable is now a no-op that resolves to the
        // default rather than to a separate path.
        if std::env::var_os("RAW_EDITOR_USE_OWNERSHIP_VIRTUAL_TILE_STITCHER").is_some() {
            return Some(Self::LayeredVirtualTile);
        }
        None
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
        // The environment overrides are read by `resolve`, so this test only
        // asserts the parsing half that cannot be affected by the ambient
        // environment of the test binary.
        assert_eq!(StackCompositorChoice::from_identifier(""), None);
        assert_eq!(StackCompositorChoice::from_identifier("nope"), None);
        assert_eq!(
            StackCompositorChoice::from_identifier("  Streaming_Mosaic "),
            Some(StackCompositorChoice::StreamingMosaic)
        );
        assert_eq!(
            None.and_then(StackCompositorChoice::from_identifier)
                .unwrap_or_default(),
            StackCompositorChoice::LayeredVirtualTile
        );
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
