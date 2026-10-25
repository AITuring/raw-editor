//! Configuration and finishing modes for the ownership compositor.
//!
//! The compositor itself remains in `stitching.rs` because it owns the full
//! resolution sampling loop.  These small, pure values are kept separately so
//! the entry points can share one explicit finishing contract without mixing
//! environment parsing into the renderer.

/// How the ownership compositor finishes a composed canvas.
///
/// The two arms exist so the retired path keeps its exact pixels while the
/// default path drops the second-pass sharpen and the rectangle crop.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum TileCompositorFinishing {
    /// Comparison path: 0.85 final sharpen plus the valid-rectangle crop.
    LegacyOwnership,
    /// Default path: no sharpen, no crop beyond the covered union bounds.
    LayeredVirtualTile,
}

/// The final sharpen amount of the default layered path (任务 13.4).
///
/// 需求 11.6 / 11.7 measure sharpness with MTF50 and gradient energy, so the
/// default path must not pre-sharpen its own evidence: the default is 0.0.
/// `RAW_EDITOR_FINAL_PANORAMA_SHARPEN_AMOUNT` survives as a diagnostic knob
/// and the resolved value is reported in `composition.final_sharpen_amount`.
pub(crate) fn layered_final_sharpen_amount() -> f32 {
    std::env::var("RAW_EDITOR_FINAL_PANORAMA_SHARPEN_AMOUNT")
        .ok()
        .and_then(|value| value.parse::<f32>().ok())
        .filter(|value| value.is_finite())
        .unwrap_or(0.0)
        .clamp(0.0, 1.25)
}

/// Diagnostic switches of the ownership compositor. The retired comparison
/// path reads them once (`from_env`); the default layered path always uses
/// `Default`, all off, so no environment variable changes its output
/// (需求 15.2 / 15.10, 任务 13.7).
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct OwnershipCompositorSwitches {
    pub(super) tone_all_pixels: bool,
    pub(super) soft_owner_boundary: bool,
    pub(super) skip_exposure_gain: bool,
    pub(super) low_frequency_consensus: bool,
    pub(super) owner_tone_harmonization: bool,
    pub(super) skip_owner_tone_harmonization: bool,
    pub(super) final_low_frequency_illumination: bool,
    pub(super) legacy_final_sharpen_amount: f32,
}

impl Default for OwnershipCompositorSwitches {
    fn default() -> Self {
        Self {
            tone_all_pixels: false,
            soft_owner_boundary: false,
            skip_exposure_gain: false,
            low_frequency_consensus: false,
            owner_tone_harmonization: false,
            skip_owner_tone_harmonization: false,
            final_low_frequency_illumination: false,
            legacy_final_sharpen_amount: 0.85,
        }
    }
}

impl OwnershipCompositorSwitches {
    fn from_env() -> Self {
        let set = |name: &str| std::env::var_os(name).is_some();
        Self {
            tone_all_pixels: set("RAW_EDITOR_FOCUS_TONE_ALL_PIXELS"),
            soft_owner_boundary: set("RAW_EDITOR_FOCUS_SOFT_OWNER_BOUNDARY"),
            skip_exposure_gain: set("RAW_EDITOR_SKIP_FOCUS_TILE_EXPOSURE_GAIN"),
            low_frequency_consensus: set("RAW_EDITOR_ENABLE_FOCUS_TILE_LOW_FREQUENCY_CONSENSUS"),
            owner_tone_harmonization: set("RAW_EDITOR_ENABLE_FOCUS_TILE_OWNER_TONE_HARMONIZATION"),
            skip_owner_tone_harmonization: set("RAW_EDITOR_SKIP_FOCUS_TONE_HARMONIZATION"),
            final_low_frequency_illumination: set(
                "RAW_EDITOR_ENABLE_FINAL_LOW_FREQUENCY_ILLUMINATION",
            ),
            legacy_final_sharpen_amount: std::env::var("RAW_EDITOR_FINAL_FOCUS_SHARPEN_AMOUNT")
                .ok()
                .and_then(|value| value.parse::<f32>().ok())
                .filter(|value| value.is_finite())
                .unwrap_or(0.85)
                .clamp(0.0, 1.25),
        }
    }
}

/// Resolve the finishing mode's switches at the production entry point.
/// Keeping this decision in one function lets the layered-path test exercise
/// the same branch used by the compositor, rather than reconstructing it in a
/// test-only helper.
pub(super) fn ownership_compositor_switches(
    finishing: TileCompositorFinishing,
) -> OwnershipCompositorSwitches {
    match finishing {
        TileCompositorFinishing::LegacyOwnership => OwnershipCompositorSwitches::from_env(),
        TileCompositorFinishing::LayeredVirtualTile => OwnershipCompositorSwitches::default(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_ownership_switches_are_all_off_for_the_layered_path() {
        let switches = ownership_compositor_switches(TileCompositorFinishing::LayeredVirtualTile);
        assert!(!switches.tone_all_pixels);
        assert!(!switches.soft_owner_boundary);
        assert!(!switches.skip_exposure_gain);
        assert!(!switches.low_frequency_consensus);
        assert!(!switches.owner_tone_harmonization);
        assert!(!switches.skip_owner_tone_harmonization);
        assert!(!switches.final_low_frequency_illumination);
        assert_eq!(switches.legacy_final_sharpen_amount, 0.85);
        assert_eq!(
            crate::panorama_utils::stack_pipeline::report::SelectedPath::LegacySingleLayerMosaic
                .as_identifier(),
            "legacy_single_layer_mosaic"
        );
    }

    #[test]
    fn layered_entry_uses_default_switches_while_legacy_entry_keeps_its_diagnostic_defaults() {
        let layered = ownership_compositor_switches(TileCompositorFinishing::LayeredVirtualTile);
        let legacy = ownership_compositor_switches(TileCompositorFinishing::LegacyOwnership);

        assert_eq!(layered, OwnershipCompositorSwitches::default());
        assert_eq!(legacy.legacy_final_sharpen_amount, 0.85);
        assert!(!legacy.tone_all_pixels);
        assert!(!legacy.soft_owner_boundary);
        assert!(!legacy.skip_exposure_gain);
    }
}
