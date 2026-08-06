use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum RenderTier {
    RapidPreview,
    HalfResolutionEdit,
    FullResolutionRoi,
    FullResolutionExport,
}

impl RenderTier {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::RapidPreview => "rapidPreview",
            Self::HalfResolutionEdit => "halfResolutionEdit",
            Self::FullResolutionRoi => "fullResolutionRoi",
            Self::FullResolutionExport => "fullResolutionExport",
        }
    }
}

pub fn resolve_preview_render_tier(
    requested: Option<RenderTier>,
    is_interactive: bool,
    has_roi: bool,
    target_resolution: u32,
    source_longest_edge: u32,
) -> Result<RenderTier, String> {
    let fallback = if is_interactive {
        RenderTier::RapidPreview
    } else if has_roi && source_longest_edge > 0 && target_resolution >= source_longest_edge {
        RenderTier::FullResolutionRoi
    } else {
        RenderTier::HalfResolutionEdit
    };
    let tier = requested.unwrap_or(fallback);

    match tier {
        RenderTier::RapidPreview if !is_interactive => {
            Err("rapidPreview is only valid for interactive preview jobs".to_string())
        }
        RenderTier::HalfResolutionEdit if is_interactive => {
            Err("interactive preview jobs must use rapidPreview".to_string())
        }
        RenderTier::FullResolutionRoi if is_interactive => {
            Err("fullResolutionRoi is only valid for settled preview jobs".to_string())
        }
        RenderTier::FullResolutionRoi if !has_roi => {
            Err("fullResolutionRoi requires a viewport ROI".to_string())
        }
        RenderTier::FullResolutionRoi
            if source_longest_edge > 0 && target_resolution < source_longest_edge =>
        {
            Err("fullResolutionRoi must render at the source resolution".to_string())
        }
        RenderTier::FullResolutionExport => {
            Err("fullResolutionExport cannot be submitted to the preview worker".to_string())
        }
        _ => Ok(tier),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MaskTexturePlan {
    pub width: u32,
    pub height: u32,
    pub layers: u32,
    pub upload_layers: u32,
    pub use_dummy: bool,
}

impl MaskTexturePlan {
    pub fn new(
        width: u32,
        height: u32,
        active_mask_count: usize,
        available_bitmap_count: usize,
        max_masks: usize,
    ) -> Self {
        let layers = active_mask_count.max(available_bitmap_count).min(max_masks) as u32;
        let upload_layers = available_bitmap_count.min(layers as usize) as u32;
        if layers == 0 {
            Self {
                width: 1,
                height: 1,
                layers: 1,
                upload_layers: 0,
                use_dummy: true,
            }
        } else {
            Self {
                width,
                height,
                layers,
                upload_layers,
                use_dummy: false,
            }
        }
    }

    pub fn logical_texture_bytes(self) -> usize {
        (self.width as usize)
            .saturating_mul(self.height as usize)
            .saturating_mul(self.layers as usize)
    }
}

#[cfg(test)]
mod tests {
    use super::{MaskTexturePlan, RenderTier, resolve_preview_render_tier};

    const A7R_V_WIDTH: u32 = 9_504;
    const A7R_V_HEIGHT: u32 = 6_336;

    #[test]
    fn render_tiers_use_the_camel_case_ipc_contract() {
        let tiers = [
            (RenderTier::RapidPreview, "\"rapidPreview\""),
            (RenderTier::HalfResolutionEdit, "\"halfResolutionEdit\""),
            (RenderTier::FullResolutionRoi, "\"fullResolutionRoi\""),
            (RenderTier::FullResolutionExport, "\"fullResolutionExport\""),
        ];

        for (tier, encoded) in tiers {
            assert_eq!(serde_json::to_string(&tier).unwrap(), encoded);
            assert_eq!(serde_json::from_str::<RenderTier>(encoded).unwrap(), tier);
        }
    }

    #[test]
    fn preview_tiers_reject_crossing_the_preview_export_boundary() {
        assert_eq!(
            resolve_preview_render_tier(None, true, false, 2_048, A7R_V_WIDTH).unwrap(),
            RenderTier::RapidPreview
        );
        assert_eq!(
            resolve_preview_render_tier(None, false, true, A7R_V_WIDTH, A7R_V_WIDTH,).unwrap(),
            RenderTier::FullResolutionRoi
        );
        assert_eq!(
            resolve_preview_render_tier(
                Some(RenderTier::RapidPreview),
                true,
                false,
                2_048,
                A7R_V_WIDTH,
            )
            .unwrap(),
            RenderTier::RapidPreview
        );
        assert_eq!(
            resolve_preview_render_tier(
                Some(RenderTier::FullResolutionRoi),
                false,
                true,
                A7R_V_WIDTH,
                A7R_V_WIDTH,
            )
            .unwrap(),
            RenderTier::FullResolutionRoi
        );
        assert!(
            resolve_preview_render_tier(
                Some(RenderTier::FullResolutionExport),
                false,
                false,
                A7R_V_WIDTH,
                A7R_V_WIDTH,
            )
            .is_err()
        );
        assert!(
            resolve_preview_render_tier(
                Some(RenderTier::FullResolutionRoi),
                false,
                true,
                A7R_V_WIDTH / 2,
                A7R_V_WIDTH,
            )
            .is_err()
        );
    }

    #[test]
    fn empty_masks_use_a_constant_dummy_texture_at_60mp() {
        let plan = MaskTexturePlan::new(A7R_V_WIDTH, A7R_V_HEIGHT, 0, 0, 32);
        assert!(plan.use_dummy);
        assert_eq!(plan.upload_layers, 0);
        assert_eq!(plan.logical_texture_bytes(), 1);

        let legacy_two_layer_bytes = A7R_V_WIDTH as usize * A7R_V_HEIGHT as usize * 2;
        assert_eq!(legacy_two_layer_bytes, 120_434_688);

        let one_mask = MaskTexturePlan::new(A7R_V_WIDTH, A7R_V_HEIGHT, 1, 1, 32);
        assert_eq!(one_mask.upload_layers, 1);
        assert_eq!(one_mask.logical_texture_bytes(), 60_217_344);

        let missing_bitmap = MaskTexturePlan::new(A7R_V_WIDTH, A7R_V_HEIGHT, 2, 1, 32);
        assert!(!missing_bitmap.use_dummy);
        assert_eq!(missing_bitmap.layers, 2);
        assert_eq!(missing_bitmap.upload_layers, 1);

        let all_missing = MaskTexturePlan::new(A7R_V_WIDTH, A7R_V_HEIGHT, 2, 0, 32);
        assert!(!all_missing.use_dummy);
        assert_eq!(all_missing.layers, 2);
        assert_eq!(all_missing.upload_layers, 0);

        let capped = MaskTexturePlan::new(A7R_V_WIDTH, A7R_V_HEIGHT, 40, 40, 32);
        assert_eq!(capped.upload_layers, 32);
    }
}
