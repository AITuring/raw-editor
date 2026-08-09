use serde::{Deserialize, Serialize};

const MIB: usize = 1024 * 1024;

pub const GPU_TILE_SIZE: u32 = 2_048;
pub const GPU_TILE_OVERLAP: u32 = 128;
pub const GPU_RECLAIM_HIGH_WATER_BYTES: usize = 512 * MIB;

pub fn should_reclaim_gpu_resources(processor_bytes: usize, input_bytes: usize) -> bool {
    processor_bytes.saturating_add(input_bytes) >= GPU_RECLAIM_HIGH_WATER_BYTES
}

const RGBA8_BYTES_PER_PIXEL: usize = 4;
const RGBA16_FLOAT_BYTES_PER_PIXEL: usize = 8;
const PROCESSOR_RGBA16_TILE_COUNT: usize = 5;
const FLARE_TEXTURE_COUNT: usize = 3;
const FLARE_TEXTURE_EDGE: usize = 512;

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

/// Logical sizes of the long-lived textures owned by a `GpuProcessor`.
///
/// CPU readback jobs only need the reusable tile textures. Native display jobs
/// additionally need two full-frame RGBA8 surfaces. Keeping those two cases
/// explicit prevents a 60 MP file export from retaining blank full-frame
/// display textures after the encoder has finished.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GpuProcessorTexturePlan {
    pub processing_width: u32,
    pub processing_height: u32,
    pub display_width: u32,
    pub display_height: u32,
}

impl GpuProcessorTexturePlan {
    pub fn new(width: u32, height: u32, needs_native_display: bool) -> Self {
        let processing_width = width.saturating_add(255) & !255;
        let processing_height = height.saturating_add(255) & !255;
        Self {
            processing_width: processing_width.max(1),
            processing_height: processing_height.max(1),
            display_width: if needs_native_display {
                processing_width.max(1)
            } else {
                1
            },
            display_height: if needs_native_display {
                processing_height.max(1)
            } else {
                1
            },
        }
    }

    pub fn has_native_display_surfaces(self) -> bool {
        self.display_width > 1 || self.display_height > 1
    }

    pub fn logical_texture_bytes(self) -> usize {
        let tile_width = self
            .processing_width
            .min(GPU_TILE_SIZE + GPU_TILE_OVERLAP * 2) as usize;
        let tile_height = self
            .processing_height
            .min(GPU_TILE_SIZE + GPU_TILE_OVERLAP * 2) as usize;
        let tile_pixels = tile_width.saturating_mul(tile_height);
        let tile_bytes = tile_pixels.saturating_mul(
            PROCESSOR_RGBA16_TILE_COUNT * RGBA16_FLOAT_BYTES_PER_PIXEL + RGBA8_BYTES_PER_PIXEL,
        );
        let display_bytes = (self.display_width as usize)
            .saturating_mul(self.display_height as usize)
            .saturating_mul(RGBA8_BYTES_PER_PIXEL * 2);
        let flare_bytes = FLARE_TEXTURE_EDGE
            .saturating_mul(FLARE_TEXTURE_EDGE)
            .saturating_mul(FLARE_TEXTURE_COUNT)
            .saturating_mul(RGBA16_FLOAT_BYTES_PER_PIXEL);

        tile_bytes
            .saturating_add(display_bytes)
            .saturating_add(flare_bytes)
    }

    pub fn should_rebuild_for(self, width: u32, height: u32, needs_native_display: bool) -> bool {
        let requested = Self::new(width, height, needs_native_display);
        if self.processing_width < width
            || self.processing_height < height
            || (needs_native_display
                && (self.display_width < width || self.display_height < height))
        {
            return true;
        }

        if !needs_native_display
            && self.has_native_display_surfaces()
            && self.logical_texture_bytes() >= GPU_RECLAIM_HIGH_WATER_BYTES
            && requested.logical_texture_bytes().saturating_mul(2) <= self.logical_texture_bytes()
        {
            return true;
        }

        let current_area =
            (self.processing_width as usize).saturating_mul(self.processing_height as usize);
        let requested_area = (requested.processing_width as usize)
            .saturating_mul(requested.processing_height as usize);
        self.logical_texture_bytes() >= GPU_RECLAIM_HIGH_WATER_BYTES
            && requested_area.saturating_mul(4) <= current_area
            && requested.logical_texture_bytes().saturating_mul(2) <= self.logical_texture_bytes()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StreamingExportBufferPlan {
    pub width: u32,
    pub height: u32,
    pub band_rows: u32,
}

impl StreamingExportBufferPlan {
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            band_rows: height.min(GPU_TILE_SIZE),
        }
    }

    pub fn band_rgba_bytes(self) -> usize {
        (self.width as usize)
            .saturating_mul(self.band_rows as usize)
            .saturating_mul(RGBA8_BYTES_PER_PIXEL)
    }

    pub fn legacy_full_rgba_bytes(self) -> usize {
        (self.width as usize)
            .saturating_mul(self.height as usize)
            .saturating_mul(RGBA8_BYTES_PER_PIXEL)
    }
}

#[cfg(test)]
mod tests {
    use super::{
        GPU_RECLAIM_HIGH_WATER_BYTES, GpuProcessorTexturePlan, MaskTexturePlan, RenderTier,
        StreamingExportBufferPlan, resolve_preview_render_tier, should_reclaim_gpu_resources,
    };

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

    #[test]
    fn cpu_export_avoids_full_frame_display_textures() {
        let cpu_export = GpuProcessorTexturePlan::new(A7R_V_WIDTH, A7R_V_HEIGHT, false);
        let native_display = GpuProcessorTexturePlan::new(A7R_V_WIDTH, A7R_V_HEIGHT, true);

        assert!(!cpu_export.has_native_display_surfaces());
        assert!(native_display.has_native_display_surfaces());
        assert_eq!(cpu_export.display_width, 1);
        assert_eq!(cpu_export.display_height, 1);
        assert_eq!(native_display.display_width, 9_728);
        assert_eq!(native_display.display_height, 6_400);
        assert_eq!(
            native_display.logical_texture_bytes() - cpu_export.logical_texture_bytes(),
            498_073_592
        );
    }

    #[test]
    fn high_water_processor_shrinks_with_hysteresis() {
        let high_water = GpuProcessorTexturePlan::new(A7R_V_WIDTH, A7R_V_HEIGHT, true);
        assert!(high_water.logical_texture_bytes() > GPU_RECLAIM_HIGH_WATER_BYTES);
        assert!(high_water.should_rebuild_for(A7R_V_WIDTH, A7R_V_HEIGHT, false));
        assert!(high_water.should_rebuild_for(2_048, 1_365, true));
        assert!(!high_water.should_rebuild_for(A7R_V_WIDTH, A7R_V_HEIGHT, true));

        let ordinary_preview = GpuProcessorTexturePlan::new(2_048, 1_365, true);
        assert!(!ordinary_preview.should_rebuild_for(1_920, 1_280, true));
        assert!(ordinary_preview.should_rebuild_for(4_096, 2_731, true));
    }

    #[test]
    fn gpu_reclamation_uses_the_combined_processor_and_input_high_water() {
        assert!(!should_reclaim_gpu_resources(
            256 * 1024 * 1024,
            255 * 1024 * 1024
        ));
        assert!(should_reclaim_gpu_resources(
            256 * 1024 * 1024,
            256 * 1024 * 1024
        ));
        assert!(should_reclaim_gpu_resources(usize::MAX, usize::MAX));
    }

    #[test]
    fn streaming_export_bounds_the_60mp_cpu_output_band() {
        let plan = StreamingExportBufferPlan::new(A7R_V_WIDTH, A7R_V_HEIGHT);
        assert_eq!(plan.band_rows, 2_048);
        assert_eq!(plan.band_rgba_bytes(), 77_856_768);
        assert_eq!(plan.legacy_full_rgba_bytes(), 240_869_376);
        assert_eq!(
            plan.legacy_full_rgba_bytes() - plan.band_rgba_bytes(),
            163_012_608
        );
    }
}
