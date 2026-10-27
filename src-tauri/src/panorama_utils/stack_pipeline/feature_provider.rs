//! Stable boundary types for optional proxy-resolution feature providers.
//!
//! The production path remains the native FAST/BRIEF implementation in
//! `panorama_stitching`.  This module deliberately contains no Python, OpenCV,
//! PyO3, or maturin dependency.  It defines the small value schema that a
//! future optional provider (or an offline sidecar) must satisfy before Rust
//! is allowed to run matching and geometry verification.

use serde::{Deserialize, Serialize};

/// Version of the feature-provider value schema.
pub(crate) const FEATURE_PROVIDER_SCHEMA: u32 = 1;
/// Stable identifier of the default, always available provider.
pub(crate) const RUST_FAST_BRIEF_PROVIDER: &str = "rust_fast_brief";
/// Stable identifier reserved for the optional OpenCV SIFT provider.
pub(crate) const PYTHON_OPENCV_SIFT_PROVIDER: &str = "python_opencv_sift";
/// Stable identifier reserved for the optional OpenCV SuperPoint provider.
pub(crate) const PYTHON_OPENCV_SUPERPOINT_PROVIDER: &str = "python_opencv_superpoint";

/// A provider name is serialized instead of relying on a Rust enum spelling.
/// That keeps reports and sidecar payloads forward-compatible with providers
/// added after the first schema version.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum FeatureProviderKind {
    RustFastBrief,
    PythonOpencvSift,
    PythonOpencvSuperpoint,
}

impl FeatureProviderKind {
    pub(crate) const fn identifier(self) -> &'static str {
        match self {
            Self::RustFastBrief => RUST_FAST_BRIEF_PROVIDER,
            Self::PythonOpencvSift => PYTHON_OPENCV_SIFT_PROVIDER,
            Self::PythonOpencvSuperpoint => PYTHON_OPENCV_SUPERPOINT_PROVIDER,
        }
    }

    pub(crate) const fn is_optional_python(self) -> bool {
        matches!(self, Self::PythonOpencvSift | Self::PythonOpencvSuperpoint)
    }
}

/// Coordinate frame attached to every feature payload.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct CoordinateFrame {
    pub width: u32,
    pub height: u32,
    pub scale_x: f64,
    pub scale_y: f64,
    pub coordinate_system_id: String,
}

impl CoordinateFrame {
    pub(crate) fn new(
        width: u32,
        height: u32,
        scale_x: f64,
        scale_y: f64,
        coordinate_system_id: impl Into<String>,
    ) -> Self {
        Self {
            width,
            height,
            scale_x,
            scale_y,
            coordinate_system_id: coordinate_system_id.into(),
        }
    }

    fn validate(&self) -> Result<(), FeatureProviderError> {
        if self.width == 0 || self.height == 0 {
            return Err(FeatureProviderError::InvalidResult {
                detail: "feature coordinate frame has zero dimensions".to_string(),
            });
        }
        if !self.scale_x.is_finite()
            || !self.scale_y.is_finite()
            || self.scale_x <= 0.0
            || self.scale_y <= 0.0
        {
            return Err(FeatureProviderError::InvalidResult {
                detail: "feature coordinate frame has invalid scale".to_string(),
            });
        }
        if self.coordinate_system_id.trim().is_empty() {
            return Err(FeatureProviderError::InvalidResult {
                detail: "feature coordinate frame has no coordinate_system_id".to_string(),
            });
        }
        Ok(())
    }
}

/// A compact keypoint. Coordinates are in the attached proxy frame.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub(crate) struct FeaturePoint {
    pub x: f32,
    pub y: f32,
    pub scale: f32,
    pub orientation: f32,
    pub response: f32,
}

impl FeaturePoint {
    fn validate(&self, frame: &CoordinateFrame) -> Result<(), FeatureProviderError> {
        if !self.x.is_finite()
            || !self.y.is_finite()
            || !self.scale.is_finite()
            || !self.orientation.is_finite()
            || !self.response.is_finite()
            || self.scale <= 0.0
            || self.x < 0.0
            || self.y < 0.0
            || self.x >= frame.width as f32
            || self.y >= frame.height as f32
        {
            return Err(FeatureProviderError::InvalidResult {
                detail: "feature point is non-finite or outside the proxy frame".to_string(),
            });
        }
        Ok(())
    }
}

/// Element encoding for one descriptor row. `dimension` is the logical
/// dimension (256 bits for BRIEF, 128 float values for SIFT), while `bytes`
/// always carries the compact wire representation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum DescriptorElementType {
    Binary,
    Uint8,
    Float32,
}

/// One descriptor row. Keeping rows separate allows a sidecar to use a
/// compact binary representation while retaining an unambiguous row count.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct Descriptor {
    pub kind: String,
    pub element_type: DescriptorElementType,
    pub dimension: u32,
    pub bytes: Vec<u8>,
}

impl Descriptor {
    fn validate(&self) -> Result<(), FeatureProviderError> {
        if self.kind.trim().is_empty() || self.dimension == 0 {
            return Err(FeatureProviderError::InvalidResult {
                detail: "descriptor kind or dimension is empty".to_string(),
            });
        }
        let expected_bytes = match self.element_type {
            DescriptorElementType::Binary => self.dimension.div_ceil(8) as usize,
            DescriptorElementType::Uint8 => self.dimension as usize,
            DescriptorElementType::Float32 => self.dimension.saturating_mul(4) as usize,
        };
        if self.bytes.len() != expected_bytes {
            return Err(FeatureProviderError::InvalidResult {
                detail: format!(
                    "descriptor byte length {} does not equal encoded dimension {}",
                    self.bytes.len(),
                    expected_bytes
                ),
            });
        }
        Ok(())
    }
}

/// A compact one-to-one match. `confidence` is normalized to [0, 1].
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub(crate) struct FeatureMatch {
    pub source_index: u32,
    pub target_index: u32,
    pub distance: f32,
    pub confidence: f32,
}

impl FeatureMatch {
    fn validate(&self, source_len: usize, target_len: usize) -> Result<(), FeatureProviderError> {
        if self.source_index as usize >= source_len || self.target_index as usize >= target_len {
            return Err(FeatureProviderError::InvalidResult {
                detail: "feature match index is outside the keypoint arrays".to_string(),
            });
        }
        if !self.distance.is_finite()
            || self.distance < 0.0
            || !self.confidence.is_finite()
            || !(0.0..=1.0).contains(&self.confidence)
        {
            return Err(FeatureProviderError::InvalidResult {
                detail: "feature match has invalid distance or confidence".to_string(),
            });
        }
        Ok(())
    }
}

/// Value returned by every provider. Rust owns all validation after this
/// boundary; no provider is allowed to inject a homography or skip RANSAC.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct FeatureProviderResult {
    pub schema: u32,
    pub provider: FeatureProviderKind,
    pub provider_version: String,
    pub frame: CoordinateFrame,
    pub keypoints: Vec<FeaturePoint>,
    pub descriptors: Vec<Descriptor>,
    pub matches: Vec<FeatureMatch>,
    /// Bytes copied across a language/process boundary. Native providers use
    /// zero; optional sidecars must report the explicit payload size.
    pub explicit_copy_bytes: u64,
}

impl FeatureProviderResult {
    /// Validate all untrusted fields and deterministic ordering. Geometry code
    /// must call this before consuming an optional provider result.
    pub(crate) fn validate(&self) -> Result<(), FeatureProviderError> {
        if self.schema != FEATURE_PROVIDER_SCHEMA {
            return Err(FeatureProviderError::InvalidResult {
                detail: format!(
                    "unsupported feature-provider schema {} (expected {})",
                    self.schema, FEATURE_PROVIDER_SCHEMA
                ),
            });
        }
        if self.provider_version.trim().is_empty() {
            return Err(FeatureProviderError::InvalidResult {
                detail: "provider_version is empty".to_string(),
            });
        }
        self.frame.validate()?;
        if self.descriptors.len() != self.keypoints.len() {
            return Err(FeatureProviderError::InvalidResult {
                detail: "descriptor and keypoint counts differ".to_string(),
            });
        }
        for point in &self.keypoints {
            point.validate(&self.frame)?;
        }
        for descriptor in &self.descriptors {
            descriptor.validate()?;
        }
        let mut previous = None;
        for matched in &self.matches {
            matched.validate(self.keypoints.len(), self.keypoints.len())?;
            let order = (matched.source_index, matched.target_index);
            if previous.is_some_and(|prior| prior > order) {
                return Err(FeatureProviderError::InvalidResult {
                    detail: "feature matches are not sorted by source/target index".to_string(),
                });
            }
            previous = Some(order);
        }
        Ok(())
    }

    /// Stable sort key for provider output. Equal keypoints are resolved by
    /// coordinates and then descriptor bytes, never by provider iteration order.
    pub(crate) fn sort_key(
        point: &FeaturePoint,
        descriptor: &Descriptor,
    ) -> (u32, u32, u32, Vec<u8>) {
        (
            point.x.to_bits(),
            point.y.to_bits(),
            point.response.to_bits(),
            descriptor.bytes.clone(),
        )
    }
}

/// Stable failure codes used for optional-provider fallback diagnostics.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum FeatureProviderError {
    PythonUnavailable,
    StartupFailed,
    AbiMismatch,
    OpencvUnavailable,
    RuntimeError,
    Cancelled,
    InvalidResult { detail: String },
    BudgetExceeded,
}

impl FeatureProviderError {
    pub(crate) const fn code(&self) -> &'static str {
        match self {
            Self::PythonUnavailable => "python_unavailable",
            Self::StartupFailed => "startup_failed",
            Self::AbiMismatch => "abi_mismatch",
            Self::OpencvUnavailable => "opencv_unavailable",
            Self::RuntimeError => "runtime_error",
            Self::Cancelled => "cancelled",
            Self::InvalidResult { .. } => "invalid_result",
            Self::BudgetExceeded => "budget_exceeded",
        }
    }
}

/// Result of resolving a requested provider. The optional Python providers are
/// intentionally unavailable in the desktop binary until a separately
/// packaged sidecar is installed; this preserves the Rust default path.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct ProviderResolution {
    pub requested: FeatureProviderKind,
    pub selected: FeatureProviderKind,
    pub fallback_reason: Option<FeatureProviderError>,
}

impl ProviderResolution {
    pub(crate) fn default_rust() -> Self {
        Self {
            requested: FeatureProviderKind::RustFastBrief,
            selected: FeatureProviderKind::RustFastBrief,
            fallback_reason: None,
        }
    }

    pub(crate) fn resolve(requested: FeatureProviderKind) -> Self {
        if requested.is_optional_python() {
            Self {
                requested,
                selected: FeatureProviderKind::RustFastBrief,
                fallback_reason: Some(FeatureProviderError::PythonUnavailable),
            }
        } else {
            Self::default_rust()
        }
    }

    /// Convert any optional-provider failure into the native provider without
    /// discarding the stable reason that must be written to Stack_Report.
    pub(crate) fn fallback(requested: FeatureProviderKind, reason: FeatureProviderError) -> Self {
        Self {
            requested,
            selected: FeatureProviderKind::RustFastBrief,
            fallback_reason: Some(reason),
        }
    }
}

/// Shared thread budget for Rayon and an optional OpenCV provider. OpenCV is
/// never allowed to create a nested pool larger than the remaining budget.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct ProviderThreadBudget {
    pub total_threads: u16,
    pub rayon_threads: u16,
    pub opencv_threads: u16,
}

impl ProviderThreadBudget {
    pub(crate) fn new(total_threads: u16, rayon_threads: u16) -> Self {
        // Keep one slot available for a provider call. A one-thread total is
        // therefore normalized to two slots; callers that do not select a
        // Python provider may still use Rayon alone outside this boundary.
        let total_threads = total_threads.max(2);
        let rayon_threads = rayon_threads.clamp(1, total_threads - 1);
        let opencv_threads = total_threads - rayon_threads;
        Self {
            total_threads,
            rayon_threads,
            opencv_threads,
        }
    }

    pub(crate) fn validate(&self) -> Result<(), FeatureProviderError> {
        if self.total_threads < 2
            || self.rayon_threads == 0
            || self.opencv_threads == 0
            || self.rayon_threads as u32 + self.opencv_threads as u32 > self.total_threads as u32
        {
            return Err(FeatureProviderError::BudgetExceeded);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::cmp::Ordering;

    use super::*;

    fn valid_result() -> FeatureProviderResult {
        FeatureProviderResult {
            schema: FEATURE_PROVIDER_SCHEMA,
            provider: FeatureProviderKind::RustFastBrief,
            provider_version: "rust-test".to_string(),
            frame: CoordinateFrame::new(64, 48, 4.0, 4.0, "proxy-v1"),
            keypoints: vec![FeaturePoint {
                x: 12.0,
                y: 10.0,
                scale: 1.0,
                orientation: 0.0,
                response: 0.5,
            }],
            descriptors: vec![Descriptor {
                kind: "brief".to_string(),
                element_type: DescriptorElementType::Binary,
                dimension: 16,
                bytes: vec![0x12, 0x34],
            }],
            matches: vec![],
            explicit_copy_bytes: 0,
        }
    }

    #[test]
    fn valid_payload_passes_and_round_trips() {
        let result = valid_result();
        result.validate().unwrap();
        let encoded = serde_json::to_string(&result).unwrap();
        let decoded: FeatureProviderResult = serde_json::from_str(&encoded).unwrap();
        assert_eq!(decoded, result);
    }

    #[test]
    fn malformed_payloads_are_rejected_before_geometry() {
        let mut result = valid_result();
        result.keypoints[0].x = f32::NAN;
        assert!(matches!(
            result.validate(),
            Err(FeatureProviderError::InvalidResult { .. })
        ));

        let mut result = valid_result();
        result.descriptors[0].bytes.push(0xff);
        assert!(matches!(
            result.validate(),
            Err(FeatureProviderError::InvalidResult { .. })
        ));
    }

    #[test]
    fn optional_python_resolves_to_rust_with_stable_reason() {
        let resolution = ProviderResolution::resolve(FeatureProviderKind::PythonOpencvSift);
        assert_eq!(resolution.selected, FeatureProviderKind::RustFastBrief);
        assert_eq!(
            resolution
                .fallback_reason
                .as_ref()
                .map(FeatureProviderError::code),
            Some("python_unavailable")
        );
        assert_eq!(
            resolution.requested.identifier(),
            PYTHON_OPENCV_SIFT_PROVIDER
        );

        let crashed = ProviderResolution::fallback(
            FeatureProviderKind::PythonOpencvSift,
            FeatureProviderError::RuntimeError,
        );
        assert_eq!(
            crashed
                .fallback_reason
                .as_ref()
                .map(FeatureProviderError::code),
            Some("runtime_error")
        );
    }

    #[test]
    fn matches_must_be_sorted_and_bounded() {
        let mut result = valid_result();
        result.keypoints.push(FeaturePoint {
            x: 20.0,
            y: 11.0,
            scale: 1.0,
            orientation: 0.0,
            response: 0.4,
        });
        result.descriptors.push(Descriptor {
            kind: "brief".to_string(),
            element_type: DescriptorElementType::Binary,
            dimension: 16,
            bytes: vec![0x56, 0x78],
        });
        result.matches = vec![
            FeatureMatch {
                source_index: 1,
                target_index: 0,
                distance: 2.0,
                confidence: 0.8,
            },
            FeatureMatch {
                source_index: 0,
                target_index: 1,
                distance: 1.0,
                confidence: 0.9,
            },
        ];
        assert!(matches!(
            result.validate(),
            Err(FeatureProviderError::InvalidResult { .. })
        ));
    }

    #[test]
    fn thread_budget_keeps_opencv_inside_shared_budget() {
        let budget = ProviderThreadBudget::new(8, 6);
        assert_eq!(budget.opencv_threads, 2);
        budget.validate().unwrap();
        assert!(
            ProviderThreadBudget {
                total_threads: 1,
                rayon_threads: 1,
                opencv_threads: 1,
            }
            .validate()
            .is_err()
        );
    }

    #[test]
    fn sort_key_is_stable_for_equal_provider_output() {
        let point = FeaturePoint {
            x: 1.0,
            y: 2.0,
            scale: 1.0,
            orientation: 0.0,
            response: 0.5,
        };
        let descriptor = Descriptor {
            kind: "brief".into(),
            element_type: DescriptorElementType::Binary,
            dimension: 8,
            bytes: vec![1],
        };
        assert_eq!(
            FeatureProviderResult::sort_key(&point, &descriptor)
                .cmp(&FeatureProviderResult::sort_key(&point, &descriptor)),
            Ordering::Equal
        );
    }
}
