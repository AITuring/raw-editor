//! Shared full-resolution/source and proxy-image contracts.
//!
//! This module is deliberately metadata-only for now.  It gives the feature,
//! geometry, seam and optional focus providers one coordinate contract without
//! moving the production decoder or compositor onto a new path.  A later
//! integration can attach decoded proxy pixels to [`ProxyImage`] while keeping
//! the dimensions, scale and provenance fields unchanged.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::panorama_utils::stitching::ColorEncoding;

/// Version of the proxy dimensions and coordinate convention.
pub(crate) const PROXY_CONTRACT_VERSION: &str = "proxy-v1";
/// Default maximum proxy long side (requirement 16.2).
pub(crate) const DEFAULT_PROXY_LONG_SIDE: u32 = 2_000;
/// Conservative estimate used only for choosing a bounded proxy budget.
/// It accounts for one RGB32F proxy plane plus grayscale/mask workspace.
const PROXY_WORKING_BYTES_PER_PIXEL: u64 = 16;
/// Fallback budget when the caller has no process/memory threshold yet.  The
/// preparation stage already limits concurrent workers; this run-level cap is
/// therefore high enough to preserve the count-based 2000/1800/1536 analysis
/// sizes for the supported source limit instead of silently reporting a much
/// smaller proxy than the production aligner actually uses.
const DEFAULT_PROXY_MEMORY_BUDGET_BYTES: u64 = 16 * 1024 * 1024 * 1024;

/// The colour encoding shared with the decoded RAW/render pipeline.
fn color_encoding_identifier(encoding: ColorEncoding) -> &'static str {
    match encoding {
        ColorEncoding::LinearSrgb => "linear_srgb",
        ColorEncoding::DisplaySrgb => "display_srgb",
    }
}

/// Read-only metadata for one full-resolution Source_RAW.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct FullResSource {
    pub(crate) absolute_path: PathBuf,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) source_sha256: [u8; 32],
    pub(crate) coordinate_system_id: String,
    pub(crate) color_encoding: ColorEncoding,
}

impl FullResSource {
    pub(crate) fn new(
        absolute_path: PathBuf,
        width: u32,
        height: u32,
        source_sha256: [u8; 32],
        coordinate_system_id: impl Into<String>,
        color_encoding: ColorEncoding,
    ) -> Result<Self, String> {
        if width == 0 || height == 0 {
            return Err("full-resolution source dimensions must be non-zero".to_string());
        }
        let coordinate_system_id = coordinate_system_id.into();
        if coordinate_system_id.trim().is_empty() {
            return Err(
                "full-resolution source coordinate_system_id must not be empty".to_string(),
            );
        }
        Ok(Self {
            absolute_path,
            width,
            height,
            source_sha256,
            coordinate_system_id,
            color_encoding,
        })
    }

    pub(crate) fn source_sha256_hex(&self) -> String {
        hex::encode(self.source_sha256)
    }
}

/// Deterministic analysis budget shared by all proxy consumers.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct ProxyBudget {
    pub(crate) max_long_side: u32,
    pub(crate) max_pixels: u64,
    pub(crate) input_count: usize,
    pub(crate) memory_budget_bytes: u64,
    pub(crate) proxy_version: String,
    pub(crate) resize_filter: String,
}

impl ProxyBudget {
    /// Build a budget from source dimensions, input cardinality and the
    /// run-level memory threshold.  Every adjustment only lowers the default
    /// 2000-pixel long side, so a small source is never enlarged.
    pub(crate) fn for_source(
        source_width: u32,
        source_height: u32,
        input_count: usize,
        memory_budget_bytes: Option<u64>,
    ) -> Result<Self, String> {
        if source_width == 0 || source_height == 0 {
            return Err("proxy budget source dimensions must be non-zero".to_string());
        }
        let input_count = input_count.max(1);
        let memory_budget_bytes = memory_budget_bytes
            .filter(|bytes| *bytes > 0)
            .unwrap_or(DEFAULT_PROXY_MEMORY_BUDGET_BYTES);
        let source_long_side = source_width.max(source_height);
        let source_short_side = source_width.min(source_height).max(1);
        let max_pixels = (memory_budget_bytes / input_count as u64)
            .checked_div(PROXY_WORKING_BYTES_PER_PIXEL)
            .unwrap_or(1)
            .max(1);

        // If the source keeps its aspect ratio, this is the largest long side
        // whose area stays within max_pixels.  Using the actual aspect ratio is
        // what makes non-4:3 sources retain separate x/y scales.
        let memory_long_side = ((max_pixels as f64 * source_long_side as f64
            / source_short_side as f64)
            .sqrt()
            .floor() as u32)
            .max(1);
        let count_long_side = if input_count > 128 {
            1_536
        } else if input_count > 64 {
            1_800
        } else {
            DEFAULT_PROXY_LONG_SIDE
        };
        let max_long_side = DEFAULT_PROXY_LONG_SIDE
            .min(source_long_side)
            .min(count_long_side)
            .min(memory_long_side)
            .max(1);
        Ok(Self {
            max_long_side,
            max_pixels,
            input_count,
            memory_budget_bytes,
            proxy_version: PROXY_CONTRACT_VERSION.to_string(),
            resize_filter: "triangle".to_string(),
        })
    }

    pub(crate) fn dimensions_for(&self, source_width: u32, source_height: u32) -> (u32, u32) {
        if source_width == 0 || source_height == 0 {
            return (0, 0);
        }
        let source_long_side = source_width.max(source_height);
        if source_long_side <= self.max_long_side {
            return (source_width, source_height);
        }
        let scale = self.max_long_side as f64 / source_long_side as f64;
        let width = ((source_width as f64 * scale).round() as u32).max(1);
        let height = ((source_height as f64 * scale).round() as u32).max(1);
        (
            width.min(self.max_long_side),
            height.min(self.max_long_side),
        )
    }

    /// Canonical, length-delimited material for a cache key.  The caller
    /// supplies this fragment to the run/cache key builder; keeping it here
    /// prevents a provider from silently omitting proxy version or resize
    /// parameters from repeated-run identity.
    pub(crate) fn cache_key_fragment(&self) -> String {
        format!(
            "proxy_version={};max_long_side={};max_pixels={};input_count={};memory_budget_bytes={};resize_filter={}",
            self.proxy_version,
            self.max_long_side,
            self.max_pixels,
            self.input_count,
            self.memory_budget_bytes,
            self.resize_filter,
        )
    }
}

/// Continuous coordinate transform between the full-resolution source and its
/// proxy.  Coordinates use the same pixel-origin convention on both sides;
/// no fixed 4x scale is assumed.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct ProxyCoordinateTransform {
    pub(crate) full_width: u32,
    pub(crate) full_height: u32,
    pub(crate) proxy_width: u32,
    pub(crate) proxy_height: u32,
    pub(crate) scale_x: f64,
    pub(crate) scale_y: f64,
    pub(crate) coordinate_system_id: String,
}

impl ProxyCoordinateTransform {
    fn new(
        full_width: u32,
        full_height: u32,
        proxy_width: u32,
        proxy_height: u32,
        coordinate_system_id: String,
    ) -> Result<Self, String> {
        if full_width == 0 || full_height == 0 || proxy_width == 0 || proxy_height == 0 {
            return Err("proxy coordinate dimensions must be non-zero".to_string());
        }
        let scale_x = proxy_width as f64 / full_width as f64;
        let scale_y = proxy_height as f64 / full_height as f64;
        Ok(Self {
            full_width,
            full_height,
            proxy_width,
            proxy_height,
            scale_x,
            scale_y,
            coordinate_system_id,
        })
    }

    pub(crate) fn full_to_proxy(&self, x: f64, y: f64) -> Option<(f64, f64)> {
        (x.is_finite() && y.is_finite()).then_some((x * self.scale_x, y * self.scale_y))
    }

    pub(crate) fn proxy_to_full(&self, x: f64, y: f64) -> Option<(f64, f64)> {
        (x.is_finite() && y.is_finite()).then_some((x / self.scale_x, y / self.scale_y))
    }

    pub(crate) fn round_trip_error(&self, x: f64, y: f64) -> Option<f64> {
        let (proxy_x, proxy_y) = self.full_to_proxy(x, y)?;
        let (full_x, full_y) = self.proxy_to_full(proxy_x, proxy_y)?;
        Some((full_x - x).abs().max((full_y - y).abs()))
    }
}

/// Metadata shared by all proxy consumers for one Source_RAW.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct ProxyImage {
    pub(crate) source_path: PathBuf,
    pub(crate) source_sha256: String,
    pub(crate) full_width: u32,
    pub(crate) full_height: u32,
    pub(crate) proxy_width: u32,
    pub(crate) proxy_height: u32,
    pub(crate) scale_x: f64,
    pub(crate) scale_y: f64,
    pub(crate) coordinate_system_id: String,
    pub(crate) color_encoding: String,
    pub(crate) proxy_version: String,
    pub(crate) resize_filter: String,
}

impl ProxyImage {
    pub(crate) fn from_source(
        source: &FullResSource,
        budget: &ProxyBudget,
    ) -> Result<Self, String> {
        let (proxy_width, proxy_height) = budget.dimensions_for(source.width, source.height);
        let transform = ProxyCoordinateTransform::new(
            source.width,
            source.height,
            proxy_width,
            proxy_height,
            source.coordinate_system_id.clone(),
        )?;
        Ok(Self {
            source_path: source.absolute_path.clone(),
            source_sha256: source.source_sha256_hex(),
            full_width: source.width,
            full_height: source.height,
            proxy_width,
            proxy_height,
            scale_x: transform.scale_x,
            scale_y: transform.scale_y,
            coordinate_system_id: transform.coordinate_system_id,
            color_encoding: color_encoding_identifier(source.color_encoding).to_string(),
            proxy_version: budget.proxy_version.clone(),
            resize_filter: budget.resize_filter.clone(),
        })
    }

    pub(crate) fn coordinate_transform(&self) -> Result<ProxyCoordinateTransform, String> {
        ProxyCoordinateTransform::new(
            self.full_width,
            self.full_height,
            self.proxy_width,
            self.proxy_height,
            self.coordinate_system_id.clone(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn source(width: u32, height: u32) -> FullResSource {
        FullResSource::new(
            PathBuf::from("/tmp/source.NEF"),
            width,
            height,
            [0xabu8; 32],
            "source-full-v1",
            ColorEncoding::LinearSrgb,
        )
        .expect("valid source")
    }

    #[test]
    fn default_budget_caps_long_side_and_preserves_actual_axis_scales() {
        let source = source(8001, 6003);
        let budget = ProxyBudget::for_source(source.width, source.height, 10, Some(24 << 30))
            .expect("budget");
        assert_eq!(budget.max_long_side, DEFAULT_PROXY_LONG_SIDE);
        let proxy = ProxyImage::from_source(&source, &budget).expect("proxy");
        assert_eq!((proxy.proxy_width, proxy.proxy_height), (2000, 1501));
        assert_ne!(proxy.scale_x, proxy.scale_y);
        assert_eq!(proxy.source_sha256.len(), 64);
    }

    #[test]
    fn coordinate_round_trip_stays_within_quarter_full_pixel() {
        let source = source(8191, 6003);
        let budget = ProxyBudget::for_source(source.width, source.height, 10, Some(24 << 30))
            .expect("budget");
        let proxy = ProxyImage::from_source(&source, &budget).expect("proxy");
        let transform = proxy.coordinate_transform().expect("transform");
        for y in (0..source.height).step_by(257) {
            for x in (0..source.width).step_by(263) {
                assert!(
                    transform
                        .round_trip_error(x as f64, y as f64)
                        .expect("finite")
                        <= 0.25
                );
            }
        }
        assert!(
            transform.round_trip_error((source.width - 1) as f64, (source.height - 1) as f64)
                <= Some(0.25)
        );
    }

    #[test]
    fn input_count_and_memory_budget_reduce_proxy_deterministically() {
        let source = source(8000, 6000);
        let normal = ProxyBudget::for_source(source.width, source.height, 1, Some(24 << 30))
            .expect("budget");
        let many = ProxyBudget::for_source(source.width, source.height, 200, Some(24 << 30))
            .expect("budget");
        let constrained =
            ProxyBudget::for_source(source.width, source.height, 1, Some(8 << 20)).expect("budget");
        assert!(many.max_long_side < normal.max_long_side);
        assert!(constrained.max_long_side < normal.max_long_side);
        assert_eq!(many.max_long_side, 1536);
        assert_eq!(constrained.max_long_side, 836);
        assert!(
            normal
                .cache_key_fragment()
                .contains("proxy_version=proxy-v1")
        );
        assert_ne!(
            normal.cache_key_fragment(),
            constrained.cache_key_fragment()
        );
        assert_eq!(
            many.clone(),
            ProxyBudget::for_source(source.width, source.height, 200, Some(24 << 30))
                .expect("same budget")
        );
    }

    #[test]
    fn invalid_dimensions_and_coordinates_are_rejected() {
        assert!(
            FullResSource::new(
                PathBuf::from("/tmp/source.NEF"),
                0,
                100,
                [0; 32],
                "source-full-v1",
                ColorEncoding::LinearSrgb,
            )
            .is_err()
        );
        let transform =
            ProxyCoordinateTransform::new(100, 100, 50, 50, "source-full-v1".to_string())
                .expect("transform");
        assert!(transform.full_to_proxy(f64::NAN, 1.0).is_none());
        assert!(transform.proxy_to_full(1.0, f64::INFINITY).is_none());
    }
}
