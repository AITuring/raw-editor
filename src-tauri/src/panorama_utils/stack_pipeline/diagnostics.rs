//! Diagnostics_Recorder parts that work on the final output planes
//! (需求 13.2, 13.3, 13.6, 13.7, 13.8).
//!
//! * [`validate_diagnostic_roi`] decides a world ROI's validity before
//!   anything is written;
//! * [`pixel_provenance`] answers the per-pixel provenance query from the
//!   output Ownership_Map, its legend, the Coverage_Mask and the
//!   Sharpness_Confidence plane;
//! * [`export_roi_planes`] writes the ROI's output planes into the user's
//!   directory only, all or nothing.
//!
//! Every coordinate is a world native pixel; output pixel `(0, 0)` is at the
//! canvas origin, so a diagnostic ROI and the final output share one
//! coordinate system and one pixel origin.

use std::path::{Path, PathBuf};

use super::degradation;
use crate::panorama_utils::stitching::{CoverageMask, OwnershipMap};

/// Longest side of an exportable diagnostic ROI (需求 13.3 / 13.8).
pub(crate) const DIAGNOSTIC_ROI_MAX_LONG_SIDE: u32 = 4_096;

/// A world ROI: top-left corner and size in world native pixels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct DiagnosticRoi {
    pub(crate) left: i64,
    pub(crate) top: i64,
    pub(crate) width: u32,
    pub(crate) height: u32,
}

/// The reason a diagnostic ROI is refused (需求 13.8).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RoiRejection {
    /// Zero width or height.
    Empty,
    /// The long side exceeds [`DIAGNOSTIC_ROI_MAX_LONG_SIDE`].
    LongSideExceeded,
    /// The ROI lies entirely outside the final output.
    OutsideOutput,
    /// No covered output pixel lies inside the ROI.
    NoCoverage,
}

impl RoiRejection {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Empty => "roi_empty",
            Self::LongSideExceeded => "roi_long_side_exceeded",
            Self::OutsideOutput => "roi_outside_output",
            Self::NoCoverage => "roi_without_coverage",
        }
    }

    /// The user-facing error, led by the stable `diagnostics_roi_invalid`
    /// identifier and naming the reason and the ROI.
    pub(crate) fn message(self, roi: DiagnosticRoi) -> String {
        format!(
            "{}: {} for the world ROI at ({}, {}) of {}x{}",
            degradation::DIAGNOSTICS_ROI_INVALID,
            self.as_str(),
            roi.left,
            roi.top,
            roi.width,
            roi.height
        )
    }
}

/// The part of the output canvas an accepted ROI covers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct RoiWindow {
    pub(crate) x: u32,
    pub(crate) y: u32,
    pub(crate) width: u32,
    pub(crate) height: u32,
}

/// Validate a world ROI against the final output (需求 13.8). `origin` is
/// the world coordinate of output pixel `(0, 0)`. An accepted ROI returns its
/// window on the canvas, clipped to it; nothing is written either way.
pub(crate) fn validate_diagnostic_roi(
    roi: DiagnosticRoi,
    origin: (i64, i64),
    coverage: &CoverageMask,
) -> Result<RoiWindow, RoiRejection> {
    if roi.width == 0 || roi.height == 0 {
        return Err(RoiRejection::Empty);
    }
    if roi.width.max(roi.height) > DIAGNOSTIC_ROI_MAX_LONG_SIDE {
        return Err(RoiRejection::LongSideExceeded);
    }
    let (canvas_width, canvas_height) = coverage.dimensions();
    let left = roi.left - origin.0;
    let top = roi.top - origin.1;
    let x0 = left.max(0);
    let y0 = top.max(0);
    let x1 = (left + i64::from(roi.width)).min(i64::from(canvas_width));
    let y1 = (top + i64::from(roi.height)).min(i64::from(canvas_height));
    if x0 >= x1 || y0 >= y1 {
        return Err(RoiRejection::OutsideOutput);
    }
    let window = RoiWindow {
        x: x0 as u32,
        y: y0 as u32,
        width: (x1 - x0) as u32,
        height: (y1 - y0) as u32,
    };
    let covered = (window.y..window.y + window.height)
        .any(|y| (window.x..window.x + window.width).any(|x| coverage.is_covered(x, y)));
    if !covered {
        return Err(RoiRejection::NoCoverage);
    }
    Ok(window)
}

/// The provenance of one output pixel (需求 13.6).
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct PixelProvenance {
    pub(crate) station_index: usize,
    pub(crate) owner_path: PathBuf,
    pub(crate) sharpness_confidence: f32,
    pub(crate) covered: bool,
}

/// Why a provenance query has no answer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ProvenanceError {
    OutOfBounds,
    /// Uncovered or ownerless: the pixel is transparent.
    Transparent,
    /// The planes, the legend and the station table do not describe one canvas.
    InconsistentPlanes,
}

/// Look up an output pixel in the output planes. `station_of_owner[id - 1]` is
/// the Capture_Station of owner identifier `id`, parallel to the legend.
pub(crate) fn pixel_provenance(
    x: u32,
    y: u32,
    ownership: &OwnershipMap,
    station_of_owner: &[usize],
    coverage: &CoverageMask,
    confidence: &[f32],
) -> Result<PixelProvenance, ProvenanceError> {
    let (width, height) = ownership.dimensions();
    if coverage.dimensions() != (width, height)
        || confidence.len() != width as usize * height as usize
        || station_of_owner.len() != ownership.legend().len()
    {
        return Err(ProvenanceError::InconsistentPlanes);
    }
    if x >= width || y >= height {
        return Err(ProvenanceError::OutOfBounds);
    }
    let owner = ownership.owner_at(x, y);
    if owner == 0 || !coverage.is_covered(x, y) {
        return Err(ProvenanceError::Transparent);
    }
    let index = usize::from(owner) - 1;
    Ok(PixelProvenance {
        station_index: station_of_owner[index],
        owner_path: ownership.legend()[index].clone(),
        sharpness_confidence: confidence[y as usize * width as usize + x as usize],
        covered: true,
    })
}

/// Write an accepted ROI's output planes into `directory` (需求 13.3, 13.4,
/// 13.7, 13.8): `ownership.png` (16-bit owner identifiers), `coverage.png`,
/// `confidence.f32` (little-endian) and `roi.json` with the world origin, the
/// size and every owner's Capture_Station and path. The ROI is validated
/// first, so a refused ROI writes nothing. The files are written into a
/// hidden staging directory inside `directory` and published by one rename,
/// so a failed write leaves no partial result; the error names the directory.
#[allow(clippy::too_many_arguments)]
pub(crate) fn export_roi_planes(
    directory: &Path,
    roi: DiagnosticRoi,
    origin: (i64, i64),
    ownership: &OwnershipMap,
    station_of_owner: &[usize],
    coverage: &CoverageMask,
    confidence: &[f32],
) -> Result<PathBuf, String> {
    let window = validate_diagnostic_roi(roi, origin, coverage).map_err(|rejection| {
        degradation::record_run_degradation(
            degradation::DIAGNOSTICS_ROI_INVALID,
            serde_json::json!({
                "reason": rejection.as_str(),
                "left": roi.left,
                "top": roi.top,
                "width": roi.width,
                "height": roi.height,
            }),
        );
        rejection.message(roi)
    })?;
    let (width, _) = ownership.dimensions();
    if coverage.dimensions() != ownership.dimensions()
        || confidence.len() != ownership.owners().len()
        || station_of_owner.len() != ownership.legend().len()
    {
        return Err(format!(
            "{}: the output planes do not describe one canvas",
            degradation::DIAGNOSTICS_WRITE_FAILED
        ));
    }
    let failed = |error: String| {
        degradation::record_run_degradation(
            degradation::DIAGNOSTICS_WRITE_FAILED,
            serde_json::json!({
                "directory": directory.display().to_string(),
                "error": error,
            }),
        );
        format!(
            "{}: diagnostics could not be written to {}: {error}",
            degradation::DIAGNOSTICS_WRITE_FAILED,
            directory.display()
        )
    };
    if !directory.is_dir() {
        return Err(failed("the directory does not exist".to_string()));
    }
    let name = format!(
        "roi-{}-{}-{}x{}",
        roi.left, roi.top, window.width, window.height
    );
    let staging = directory.join(format!(".{name}.partial"));
    let target = directory.join(&name);
    let write = || -> Result<(), String> {
        std::fs::create_dir_all(&staging).map_err(|error| error.to_string())?;
        let index =
            |x: u32, y: u32| (window.y + y) as usize * width as usize + (window.x + x) as usize;
        let owners = image::ImageBuffer::<image::Luma<u16>, Vec<u16>>::from_fn(
            window.width,
            window.height,
            |x, y| image::Luma([ownership.owners()[index(x, y)]]),
        );
        owners
            .save(staging.join("ownership.png"))
            .map_err(|error| error.to_string())?;
        let covered = image::GrayImage::from_fn(window.width, window.height, |x, y| {
            image::Luma([coverage.covered()[index(x, y)]])
        });
        covered
            .save(staging.join("coverage.png"))
            .map_err(|error| error.to_string())?;
        let confidence_bytes = (0..window.height)
            .flat_map(|y| (0..window.width).map(move |x| (x, y)))
            .flat_map(|(x, y)| confidence[index(x, y)].to_le_bytes())
            .collect::<Vec<_>>();
        std::fs::write(staging.join("confidence.f32"), confidence_bytes)
            .map_err(|error| error.to_string())?;
        let legend = ownership
            .legend()
            .iter()
            .zip(station_of_owner)
            .enumerate()
            .map(|(index, (path, station))| {
                serde_json::json!({
                    "owner": index + 1,
                    "station_index": station,
                    "path": path.display().to_string(),
                })
            })
            .collect::<Vec<_>>();
        let metadata = serde_json::json!({
            "world_left": origin.0 + i64::from(window.x),
            "world_top": origin.1 + i64::from(window.y),
            "width": window.width,
            "height": window.height,
            "requested": {
                "left": roi.left,
                "top": roi.top,
                "width": roi.width,
                "height": roi.height,
            },
            "owners": legend,
        });
        std::fs::write(
            staging.join("roi.json"),
            serde_json::to_vec_pretty(&metadata).map_err(|error| error.to_string())?,
        )
        .map_err(|error| error.to_string())?;
        if target.exists() {
            std::fs::remove_dir_all(&target).map_err(|error| error.to_string())?;
        }
        std::fs::rename(&staging, &target).map_err(|error| error.to_string())
    };
    write().map_err(|error| {
        let _ = std::fs::remove_dir_all(&staging);
        failed(error)
    })?;
    Ok(target)
}

/// The seven per-station diagnostic items (需求 13.1).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DiagnosticItem {
    Members,
    FrameTransforms,
    ResidualField,
    OwnershipMap,
    SharpnessConfidence,
    CoverageMask,
    ToneField,
}

impl DiagnosticItem {
    pub(crate) const ALL: [Self; 7] = [
        Self::Members,
        Self::FrameTransforms,
        Self::ResidualField,
        Self::OwnershipMap,
        Self::SharpnessConfidence,
        Self::CoverageMask,
        Self::ToneField,
    ];

    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Members => "members",
            Self::FrameTransforms => "frame_transforms",
            Self::ResidualField => "residual_field",
            Self::OwnershipMap => "ownership_map",
            Self::SharpnessConfidence => "sharpness_confidence",
            Self::CoverageMask => "coverage_mask",
            Self::ToneField => "tone_field",
        }
    }
}

/// Diagnostics_Recorder of one run (需求 13.1, 13.2, 13.5, 13.7).
///
/// Disabled, it holds `None`: no buffer exists, no producer runs and nothing
/// is written (需求 13.5). Enabled, every item is written as soon as it is
/// recorded, named and listed with its Capture_Station, so no full-size
/// diagnostic buffer is retained. The first write failure stops all later
/// writes; `finish` then reports it with the target directory, while the final
/// output and the Stack_Report are left to their own writers (需求 13.7).
pub(crate) struct DiagnosticsRecorder {
    state: Option<Box<RecorderState>>,
}

struct RecorderState {
    directory: PathBuf,
    items: Vec<serde_json::Value>,
    crop: Option<(i64, i64, u32, u32)>,
    failure: Option<String>,
}

impl DiagnosticsRecorder {
    /// `directory` is the user's `stack_diagnostics.output_dir`; `None` means
    /// diagnostics are off.
    pub(crate) fn new(directory: Option<PathBuf>) -> Self {
        Self {
            state: directory.map(|directory| {
                Box::new(RecorderState {
                    directory,
                    items: Vec::new(),
                    crop: None,
                    failure: None,
                })
            }),
        }
    }

    pub(crate) fn is_enabled(&self) -> bool {
        self.state.is_some()
    }

    /// Record one item of one station. `produce` runs only while the recorder
    /// is enabled and has not failed, so a disabled run never builds the item.
    pub(crate) fn record(
        &mut self,
        station: usize,
        item: DiagnosticItem,
        extension: &str,
        produce: impl FnOnce() -> Vec<u8>,
    ) {
        let Some(state) = self.state.as_deref_mut() else {
            return;
        };
        if state.failure.is_some() {
            return;
        }
        let name = format!("station-{station:03}-{}.{extension}", item.as_str());
        match std::fs::write(state.directory.join(&name), produce()) {
            Ok(()) => state.items.push(serde_json::json!({
                "station_index": station,
                "item": item.as_str(),
                "file": name,
            })),
            Err(error) => state.failure = Some(error.to_string()),
        }
    }

    /// The final valid crop in world native pixels: its top-left corner and
    /// size (需求 13.2).
    pub(crate) fn record_crop(&mut self, left: i64, top: i64, width: u32, height: u32) {
        if let Some(state) = self.state.as_deref_mut() {
            state.crop = Some((left, top, width, height));
        }
    }

    /// Write the manifest. `Ok(None)` when diagnostics are off.
    pub(crate) fn finish(self) -> Result<Option<PathBuf>, String> {
        let Some(state) = self.state else {
            return Ok(None);
        };
        let directory = state.directory.clone();
        let failed = |error: String| {
            degradation::record_run_degradation(
                degradation::DIAGNOSTICS_WRITE_FAILED,
                serde_json::json!({
                    "directory": directory.display().to_string(),
                    "error": error,
                }),
            );
            format!(
                "{}: diagnostics could not be written to {}: {error}",
                degradation::DIAGNOSTICS_WRITE_FAILED,
                directory.display()
            )
        };
        if let Some(error) = state.failure {
            return Err(failed(error));
        }
        let manifest = serde_json::json!({
            "crop": state.crop.map(|(left, top, width, height)| serde_json::json!({
                "left": left,
                "top": top,
                "width": width,
                "height": height,
            })),
            "items": state.items,
        });
        let path = state.directory.join("diagnostics.json");
        let bytes =
            serde_json::to_vec_pretty(&manifest).map_err(|error| failed(error.to_string()))?;
        std::fs::write(&path, bytes).map_err(|error| failed(error.to_string()))?;
        Ok(Some(path))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    /// A `side × side` canvas: owner `1 + (x + y) % owners` inside a covered
    /// disc, owner 0 and uncovered outside it, confidence `x / side`.
    fn planes(side: u32, owners: u16) -> (OwnershipMap, Vec<usize>, CoverageMask, Vec<f32>) {
        let radius = f64::from(side) * 0.45;
        let centre = f64::from(side) * 0.5;
        let inside = |x: u32, y: u32| {
            (f64::from(x) + 0.5 - centre).hypot(f64::from(y) + 0.5 - centre) <= radius
        };
        let mut owner_plane = Vec::with_capacity((side * side) as usize);
        let mut covered = Vec::with_capacity((side * side) as usize);
        let mut confidence = Vec::with_capacity((side * side) as usize);
        for y in 0..side {
            for x in 0..side {
                let covered_here = inside(x, y);
                owner_plane.push(if covered_here {
                    1 + ((x + y) % u32::from(owners)) as u16
                } else {
                    0
                });
                covered.push(if covered_here { 255 } else { 0 });
                confidence.push(x as f32 / side as f32);
            }
        }
        let legend = (0..owners)
            .map(|owner| PathBuf::from(format!("/scan/station-{}/DSC_{owner:04}.NEF", owner % 3)))
            .collect::<Vec<_>>();
        let stations = (0..owners).map(|owner| usize::from(owner % 3)).collect();
        (
            OwnershipMap::new(side, side, owner_plane, legend).expect("valid ownership"),
            stations,
            CoverageMask::from_bytes(side, side, covered).expect("valid coverage"),
            confidence,
        )
    }

    proptest! {
        #![proptest_config(ProptestConfig { cases: 100, ..ProptestConfig::default() })]

        // Feature: layered-camera-group-focus-stitching, Property 84: 对于任意最终输出中的
        // 非透明像素坐标，返回的 Capture_Station 标识、owner Source_RAW 绝对路径、
        // Sharpness_Confidence 取值与 Coverage_Mask 取值都与输出 Ownership_Map 及其 owner
        // 图例在该位置的记录一致。
        //
        // **Validates: Requirements 13.6**
        #[test]
        fn property_84_pixel_provenance_matches_the_output_planes(
            side in 8u32..48,
            owners in 1u16..6,
            x in 0u32..48,
            y in 0u32..48,
        ) {
            let (ownership, stations, coverage, confidence) = planes(side, owners);
            let answer = pixel_provenance(x, y, &ownership, &stations, &coverage, &confidence);
            if x >= side || y >= side {
                prop_assert_eq!(answer, Err(ProvenanceError::OutOfBounds));
                return Ok(());
            }
            let owner = ownership.owner_at(x, y);
            if owner == 0 || !coverage.is_covered(x, y) {
                prop_assert_eq!(answer, Err(ProvenanceError::Transparent));
                return Ok(());
            }
            let answer = answer.expect("a covered, owned pixel has provenance");
            let index = usize::from(owner) - 1;
            prop_assert_eq!(answer.station_index, stations[index]);
            prop_assert_eq!(&answer.owner_path, &ownership.legend()[index]);
            prop_assert_eq!(
                answer.sharpness_confidence.to_bits(),
                confidence[(y * side + x) as usize].to_bits()
            );
            prop_assert!(answer.covered);
        }

        // Feature: layered-camera-group-focus-stitching, Property 85: 对于任意长边超过 4096 个
        // 世界坐标原生像素、完全落在最终有效裁切区域之外、或与 Coverage_Mask 已覆盖像素无
        // 交集的世界坐标 ROI，导出被拒绝、不写入任何部分结果，且错误提示指明 ROI 无效原因。
        //
        // **Validates: Requirements 13.8**
        #[test]
        fn property_85_invalid_diagnostic_rois_are_refused_without_writing(
            left in -80i64..80,
            top in -80i64..80,
            width in 0u32..5_000,
            height in 0u32..5_000,
            origin_x in -20i64..20,
            origin_y in -20i64..20,
        ) {
            let side = 32u32;
            let (ownership, stations, coverage, confidence) = planes(side, 3);
            let roi = DiagnosticRoi { left, top, width, height };
            let origin = (origin_x, origin_y);
            // Independent oracle of the three refusal conditions.
            let x0 = (left - origin_x).max(0);
            let y0 = (top - origin_y).max(0);
            let x1 = (left - origin_x + i64::from(width)).min(i64::from(side));
            let y1 = (top - origin_y + i64::from(height)).min(i64::from(side));
            let expected = if width == 0 || height == 0 {
                Err(RoiRejection::Empty)
            } else if width.max(height) > DIAGNOSTIC_ROI_MAX_LONG_SIDE {
                Err(RoiRejection::LongSideExceeded)
            } else if x0 >= x1 || y0 >= y1 {
                Err(RoiRejection::OutsideOutput)
            } else if !(y0..y1).any(|y| (x0..x1).any(|x| coverage.is_covered(x as u32, y as u32))) {
                Err(RoiRejection::NoCoverage)
            } else {
                Ok(RoiWindow {
                    x: x0 as u32,
                    y: y0 as u32,
                    width: (x1 - x0) as u32,
                    height: (y1 - y0) as u32,
                })
            };
            prop_assert_eq!(validate_diagnostic_roi(roi, origin, &coverage), expected);

            let directory = tempfile::tempdir().expect("temporary directory");
            let exported = export_roi_planes(
                directory.path(),
                roi,
                origin,
                &ownership,
                &stations,
                &coverage,
                &confidence,
            );
            let entries = std::fs::read_dir(directory.path())
                .expect("list")
                .count();
            match expected {
                Err(rejection) => {
                    let error = exported.expect_err("a refused ROI is not exported");
                    prop_assert!(error.starts_with(degradation::DIAGNOSTICS_ROI_INVALID), "{}", error);
                    prop_assert!(error.contains(rejection.as_str()), "{}", error);
                    prop_assert_eq!(entries, 0, "a refused ROI writes nothing");
                }
                Ok(window) => {
                    let target = exported.expect("an accepted ROI is exported");
                    prop_assert_eq!(entries, 1, "only the published ROI directory remains");
                    let owners = image::open(target.join("ownership.png"))
                        .expect("ownership plane")
                        .to_luma16();
                    prop_assert_eq!(owners.dimensions(), (window.width, window.height));
                    for (x, y, value) in owners.enumerate_pixels() {
                        prop_assert_eq!(
                            value[0],
                            ownership.owner_at(window.x + x, window.y + y)
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn a_missing_directory_is_a_diagnostics_write_failure_naming_it() {
        let (ownership, stations, coverage, confidence) = planes(16, 2);
        let directory = Path::new("/nonexistent/raw-editor-diagnostics");
        let roi = DiagnosticRoi {
            left: 0,
            top: 0,
            width: 16,
            height: 16,
        };
        let error = export_roi_planes(
            directory,
            roi,
            (0, 0),
            &ownership,
            &stations,
            &coverage,
            &confidence,
        )
        .expect_err("a missing directory cannot receive diagnostics");
        assert!(
            error.starts_with(degradation::DIAGNOSTICS_WRITE_FAILED),
            "{error}"
        );
        assert!(
            error.contains("/nonexistent/raw-editor-diagnostics"),
            "{error}"
        );
    }

    proptest! {
        #![proptest_config(ProptestConfig { cases: 100, ..ProptestConfig::default() })]

        // Feature: layered-camera-group-focus-stitching, Property 83: 对于任意输入，在堆栈诊断
        // 处于关闭状态时完整尺寸诊断缓冲的分配数量为 0，且不写入任何诊断文件。
        //
        // A disabled recorder never runs an item producer, so no full-size buffer is built
        // for diagnostics, and it has no directory to write to; an enabled recorder given
        // the same calls runs each producer exactly once and writes one file per item.
        //
        // **Validates: Requirements 13.5**
        #[test]
        fn property_83_disabled_diagnostics_allocate_and_write_nothing(
            calls in proptest::collection::vec((0usize..8, 0usize..7, 1usize..4096), 0..24),
        ) {
            let watched = tempfile::tempdir().expect("temporary directory");
            let produced = std::cell::Cell::new(0usize);
            let mut recorder = DiagnosticsRecorder::new(None);
            prop_assert!(!recorder.is_enabled());
            for &(station, item, bytes) in &calls {
                recorder.record(station, DiagnosticItem::ALL[item], "bin", || {
                    produced.set(produced.get() + 1);
                    vec![0u8; bytes]
                });
            }
            recorder.record_crop(-3, 4, 100, 80);
            prop_assert_eq!(produced.get(), 0, "a disabled recorder built a diagnostic buffer");
            prop_assert_eq!(recorder.finish(), Ok(None));
            prop_assert_eq!(std::fs::read_dir(watched.path()).expect("list").count(), 0);
            prop_assert_eq!(
                std::mem::size_of::<DiagnosticsRecorder>(),
                std::mem::size_of::<usize>(),
                "the disabled recorder is one null pointer"
            );

            let mut enabled = DiagnosticsRecorder::new(Some(watched.path().to_path_buf()));
            let mut distinct = std::collections::BTreeSet::new();
            for &(station, item, bytes) in &calls {
                distinct.insert((station, item));
                enabled.record(station, DiagnosticItem::ALL[item], "bin", || {
                    produced.set(produced.get() + 1);
                    vec![7u8; bytes]
                });
            }
            prop_assert_eq!(produced.get(), calls.len());
            let manifest = enabled.finish().expect("writable directory").expect("enabled");
            let written = std::fs::read_dir(watched.path()).expect("list").count();
            prop_assert_eq!(written, distinct.len() + 1, "one file per item plus the manifest");
            let manifest: serde_json::Value =
                serde_json::from_slice(&std::fs::read(manifest).expect("manifest")).expect("json");
            for entry in manifest["items"].as_array().expect("items") {
                let station = entry["station_index"].as_u64().expect("station");
                let file = entry["file"].as_str().expect("file");
                let expected_prefix = format!("station-{station:03}-");
                prop_assert!(file.starts_with(&expected_prefix));
            }
            prop_assert_eq!(manifest["crop"].is_null(), true);
        }
    }

    #[test]
    fn a_failed_diagnostic_write_stops_later_writes_and_names_the_directory() {
        let root = tempfile::tempdir().expect("temporary directory");
        let directory = root.path().join("diagnostics");
        std::fs::create_dir(&directory).expect("create");
        let mut recorder = DiagnosticsRecorder::new(Some(directory.clone()));
        recorder.record(0, DiagnosticItem::Members, "json", || b"[]".to_vec());
        recorder.record_crop(10, 20, 300, 200);
        std::fs::remove_dir_all(&directory).expect("remove the target mid-run");
        let later = std::cell::Cell::new(0usize);
        recorder.record(0, DiagnosticItem::OwnershipMap, "png", || {
            later.set(later.get() + 1);
            Vec::new()
        });
        recorder.record(1, DiagnosticItem::ToneField, "bin", || {
            later.set(later.get() + 1);
            Vec::new()
        });
        assert_eq!(
            later.get(),
            1,
            "only the failing write ran; later ones were skipped"
        );
        let error = recorder.finish().expect_err("the failure is reported");
        assert!(
            error.starts_with(degradation::DIAGNOSTICS_WRITE_FAILED),
            "{error}"
        );
        assert!(error.contains(&directory.display().to_string()), "{error}");
    }
}
