//! Focus-stack stitcher entry points and tile render records.
//!
//! Public focus-stack entries, foreground moves, source refinements, tile masks
//! and renders, plane sampling and station membership ordering.

use super::super::coordinates::Projection;
use super::super::stack_pipeline::focus_fuser;
use super::super::stack_pipeline::intra_station;
use super::super::stack_pipeline::report::FusionReport;
use super::super::stack_pipeline::report::IntraStationFrameRecord;
use super::super::stack_pipeline::report::IntraStationFrameStatus;
use super::super::stack_pipeline::station_degradation::GroupJoinEvidence;
use super::super::stack_pipeline::station_degradation::GroupJoinRejection;
use super::super::stack_pipeline::station_degradation::StationMember;
use super::super::stack_pipeline::station_degradation::group_join_rejection;
use crate::panorama_utils::shared::FocusLayerWarp;
use crate::panorama_utils::shared::ImageInfo;
use image::GrayImage;
use image::Rgb;
use image::Rgb32FImage;
use nalgebra::Matrix3;
use nalgebra::Point2;
use std::collections::HashMap;
use tauri::AppHandle;
use tauri::Runtime;

use super::{
    ConfidenceMap, CoverageMask, FocusSourceSampling, OwnershipMap,
    focus_stack_stitcher_with_margin_policy,
};

pub fn focus_stack_stitcher<R: Runtime, F>(
    images: &[&ImageInfo],
    global_homographies: &HashMap<usize, Matrix3<f64>>,
    projection: Projection,
    focus_warp: Option<&FocusLayerWarp>,
    capture_group_ids: Option<&HashMap<usize, u8>>,
    sequence_gap_aware: bool,
    app_handle: AppHandle<R>,
    progress_event: &str,
    load_image: &mut F,
) -> Result<Rgb32FImage, String>
where
    F: FnMut(&ImageInfo) -> Result<Rgb32FImage, String>,
{
    focus_stack_stitcher_with_margin_policy(
        images,
        global_homographies,
        projection,
        focus_warp,
        capture_group_ids,
        // The margin-filling form is the single-canvas comparison output: it
        // invents pixels for the projected trapezoid corners and publishes no
        // Ownership_Map, so it carries no Capture_Station identity of its own.
        0,
        sequence_gap_aware,
        true,
        app_handle,
        progress_event,
        load_image,
    )
    .map(FocusStackTileRender::into_image)
}

#[derive(Debug, Clone)]
pub(crate) struct FocusForegroundMove {
    pub(crate) delta: (i32, i32),
    pub(crate) left: u32,
    pub(crate) top: u32,
    pub(crate) destinations: GrayImage,
}

/// The Coverage_Mask and Ownership_Map the focus renderer already maintains
/// internally (需求 3.7 / 3.11 / 4.6).  They are recorded at the exact pixel
/// write sites of the hard focus selection, so publishing them does not change
/// a single pixel write.
/// Inverse sampling corrections actually applied after a member's homography.
#[derive(Debug, Clone, Default)]
pub(crate) struct FocusSourceRefinement {
    pub(crate) translation: (i32, i32),
    pub(crate) displacement: Option<crate::panorama_utils::mosaic::NativeResidualDisplacement>,
    pub(crate) source_sampling: Option<FocusSourceSampling>,
    pub(crate) foreground_move: Option<FocusForegroundMove>,
}

#[derive(Debug, Clone)]
pub(crate) struct FocusStackTileMasks {
    pub(crate) source_refinements: HashMap<usize, FocusSourceRefinement>,
    pub(crate) coverage: CoverageMask,
    pub(crate) ownership: OwnershipMap,
    /// Per-pixel Sharpness_Confidence of the owning ownership cell (需求 3.9).
    pub(crate) confidence: ConfidenceMap,
    /// Per-pixel Textured_Pixel evidence. A value of 255 means that at least
    /// one candidate observed a Sharpness_Score at or above the fixed 0.10
    /// floor in the owning cell (需求 11.12).
    pub(crate) textured: Vec<u8>,
    /// Compact winner/runner-up Sharpness_Score evidence at ownership-cell
    /// resolution.  This is report-only and is never consulted while writing
    /// the fused pixels.
    pub(crate) sharpness_evidence: focus_fuser::SharpnessEvidenceGrid,
    /// Native registration corrections that were actually applied to each
    /// fused Source_RAW layer.  The Quality_Gate uses these fields when it
    /// reconstructs a Source_RAW reference, so the reference follows the
    /// same local warp as the pixels that entered the Virtual_Tile.
    pub(crate) native_displacements:
        Vec<Option<crate::panorama_utils::mosaic::NativeResidualDisplacement>>,
    /// The Focus_Fuser observations of 需求 3.5 / 3.8 / 3.9.  `station_index` is
    /// filled by the call site that knows it.
    pub(crate) fusion: FusionReport,
}

impl FocusSourceRefinement {
    pub(crate) fn inverse_canvas_point(&self, point: Point2<f64>) -> Point2<f64> {
        let mut point = point;
        if let Some(moved) = self.foreground_move.as_ref() {
            let x = point.x.round() as i64 - i64::from(moved.left);
            let y = point.y.round() as i64 - i64::from(moved.top);
            if x >= 0
                && y >= 0
                && x < i64::from(moved.destinations.width())
                && y < i64::from(moved.destinations.height())
                && moved.destinations.get_pixel(x as u32, y as u32)[0] > 0
            {
                point.x -= f64::from(moved.delta.0);
                point.y -= f64::from(moved.delta.1);
            }
        }
        let delta = self
            .displacement
            .as_ref()
            .map(|field| field.at(point.x, point.y))
            .unwrap_or([0.0, 0.0]);
        Point2::new(
            point.x + delta[0] - f64::from(self.translation.0),
            point.y + delta[1] - f64::from(self.translation.1),
        )
    }
}

/// One focus-fused station: the pixels plus, for the unfilled Virtual_Tile
/// form, the masks that describe where they came from.  `masks` is `None`
/// whenever the render did not go through the hard focus selection (empty
/// input, degenerate bounds, the shifted-mosaic path, or the margin-filling
/// single-layer output), because those paths cannot attest per-pixel ownership.
#[derive(Debug, Clone)]
pub(crate) struct FocusStackTileRender {
    pub(crate) image: Rgb32FImage,
    pub(crate) masks: Option<FocusStackTileMasks>,
    pub(crate) sampling_origin: (f64, f64),
}

impl FocusStackTileRender {
    pub(super) fn unattributed(image: Rgb32FImage) -> Self {
        Self {
            image,
            masks: None,
            sampling_origin: (0.0, 0.0),
        }
    }

    pub(crate) fn into_image(self) -> Rgb32FImage {
        self.image
    }
}

/// Focus-fuse a station while preserving the projected validity mask.  Virtual
/// tiles use this form because their non-rectangular projective corners must
/// remain transparent until the tile-level compositor owns them, and because
/// the Virtual_Tile needs the Coverage_Mask and Ownership_Map that the fusion
/// builds on the way.
#[allow(clippy::too_many_arguments)]
pub(crate) fn focus_stack_stitcher_unfilled<R: Runtime, F>(
    images: &[&ImageInfo],
    global_homographies: &HashMap<usize, Matrix3<f64>>,
    projection: Projection,
    focus_warp: Option<&FocusLayerWarp>,
    capture_group_ids: Option<&HashMap<usize, u8>>,
    station_index: usize,
    sequence_gap_aware: bool,
    app_handle: AppHandle<R>,
    progress_event: &str,
    load_image: &mut F,
) -> Result<FocusStackTileRender, String>
where
    F: FnMut(&ImageInfo) -> Result<Rgb32FImage, String>,
{
    focus_stack_stitcher_with_margin_policy(
        images,
        global_homographies,
        projection,
        focus_warp,
        capture_group_ids,
        station_index,
        sequence_gap_aware,
        false,
        app_handle,
        progress_event,
        load_image,
    )
}

/// Nearest-neighbour read of a canvas-space position out of a layer that lives
/// at `(left, top)`, returning `None` outside the layer's validity mask.
///
/// The Sharpness_Score probes sit on a fixed grid whose spacing is a whole
/// number of native pixels, so there is nothing between samples to interpolate;
/// a bilinear read here would only low-pass the very gradients 需求 3.2 measures.
#[inline]
pub(super) fn sample_focus_plane(
    image: &Rgb32FImage,
    mask: &GrayImage,
    left: u32,
    top: u32,
    x: f64,
    y: f64,
) -> Option<Rgb<f32>> {
    let local_x = x - f64::from(left);
    let local_y = y - f64::from(top);
    if !local_x.is_finite() || !local_y.is_finite() || local_x < 0.0 || local_y < 0.0 {
        return None;
    }
    let (px, py) = (local_x.round(), local_y.round());
    if px < 0.0 || py < 0.0 || px >= f64::from(image.width()) || py >= f64::from(image.height()) {
        return None;
    }
    let (px, py) = (px as u32, py as u32);
    if mask.get_pixel(px, py).0[0] == 0 {
        return None;
    }
    Some(*image.get_pixel(px, py))
}

/// The Capture_Station members of 需求 12.1 / 12.2 as the default station path
/// can see them, together with every 需求 12.5 rejection its evidence supports.
///
/// The registration evidence is what the Intra_Station_Registrar recorded for
/// 需求 2.9 through [`intra_station::record_run_frame`].  Since the default path
/// runs the native patch refinement itself — `register_station_frame_native_layer`
/// below, which is `mosaic::refine_native_layer` over this path's own layer pose
/// — this function is what a *later* render of the same station reads, and what
/// the up-front plan of a station whose frames have already been measured is
/// built from.  A frame with a record is judged by the numbers in that record; a
/// frame without one has not been measured yet and is placed by its verified
/// global model plus the bounded integer translation of
/// `estimate_focus_layer_translation`, which has no control point field to accept
/// or reject and is therefore exactly `GlobalFallback`.  Nothing here invents a
/// substitute measurement: 需求 12.5 stays silent rather than judging a frame on a
/// number that was never taken.
pub(super) fn station_members_from_records(
    images: &[&ImageInfo],
    station_index: usize,
) -> (Vec<StationMember>, Vec<GroupJoinRejection>) {
    let recorded = intra_station::run_records_snapshot()
        .into_iter()
        .find(|station| station.station_index == station_index)
        .map(|station| {
            station
                .frames
                .into_iter()
                .map(|frame| (frame.path.clone(), frame))
                .collect::<HashMap<String, IntraStationFrameRecord>>()
        })
        .unwrap_or_default();
    station_members_from_frame_records(images, &recorded)
}

/// [`station_members_from_records`] over an injected record set.
///
/// The run scoped sink is a global that concurrent tests share, so the rule
/// itself takes the records as data: production passes the snapshot of the
/// station it is rendering, a test passes a local map.
pub(super) fn station_members_from_frame_records(
    images: &[&ImageInfo],
    recorded: &HashMap<String, IntraStationFrameRecord>,
) -> (Vec<StationMember>, Vec<GroupJoinRejection>) {
    // Layer 0 is the frame every later layer registers against, so it is this
    // station's anchor by construction and 需求 12.5 scales its error limit by
    // that frame's native long side.
    let anchor_long_side = images
        .first()
        .map(|info| info.width.max(info.height))
        .unwrap_or(0);
    let mut rejections = Vec::new();
    let members = images
        .iter()
        .enumerate()
        .map(|(index, info)| {
            let record = recorded.get(&info.filename);
            // 需求 12.5: inlier spatial support below 20% of the overlap area, or
            // an inlier median symmetric reprojection error above 0.01 × the long
            // side, keeps this Source_RAW out of every Capture_Station.  Only a
            // record with an accepted inlier set carries both measurements; with
            // none accepted the two fields are structural zeros rather than
            // measurements, and 需求 12.2's registration-failure exclusion below
            // already covers that frame.
            let rejection = record
                .filter(|record| index > 0 && record.inliers > 0)
                .and_then(|record| {
                    group_join_rejection(&GroupJoinEvidence {
                        path: info.filename.clone(),
                        inlier_area_coverage: record.inlier_area_coverage,
                        median_symmetric_error_px: record.median_symmetric_error_px,
                        anchor_long_side_px: anchor_long_side,
                    })
                });
            let status = match (
                index,
                rejection.is_some(),
                record.map(|record| record.status),
            ) {
                // Rejected from every Capture_Station, so this station cannot
                // fuse it either (需求 12.2 / 12.5).
                (_, true, _) => IntraStationFrameStatus::Failed,
                (0, _, _) => IntraStationFrameStatus::Anchor,
                (_, false, Some(status)) => status,
                (_, false, None) => IntraStationFrameStatus::GlobalFallback,
            };
            if let Some(rejection) = rejection {
                rejections.push(rejection);
            }
            StationMember {
                path: info.filename.clone(),
                median_sharpness: crate::panorama_utils::mosaic::station_member_median_sharpness(
                    info,
                ),
                status,
            }
        })
        .collect();
    (members, rejections)
}

/// Put a compact focus station into the deterministic order required by the
/// Intra_Station_Registrar: the frame with the highest valid-pixel median
/// Sharpness_Score is the anchor, and the remaining frames descend by the same
/// score with the absolute path as the stable tie-break. The grouping stage
/// orders members by world position, which is useful for topology but is not a
/// focus anchor rule; carrying that order into the compositor can make a soft
/// first frame the reference and then reject the sharper focal plane on the
/// native coverage gate.
pub(super) fn order_focus_station_frames<'a>(
    images: &'a [&'a ImageInfo],
    station_index: usize,
) -> Vec<&'a ImageInfo> {
    if images.len() < 2 {
        return images.to_vec();
    }
    let mut scored = images
        .iter()
        .copied()
        .map(|image| {
            (
                image,
                crate::panorama_utils::mosaic::station_member_median_sharpness(image),
            )
        })
        .collect::<Vec<_>>();
    // A station can be rendered more than once during one run. Once the
    // registrar has established its anchor, keep that identity stable for all
    // later renders (and for injected/property records); reselecting by a
    // recomputed sharpness score would reinterpret the existing frame records
    // against a different coordinate system. Only a station with no recorded
    // anchor takes the normal sharpness-based selection path.
    let recorded_anchor = intra_station::run_records_snapshot()
        .into_iter()
        .find(|station| station.station_index == station_index)
        .map(|station| station.anchor_path)
        .filter(|path| !path.is_empty());
    let anchor_index = recorded_anchor
        .as_deref()
        .and_then(|path| scored.iter().position(|(image, _)| image.filename == path))
        .or_else(|| {
            let candidates = scored
                .iter()
                .map(|(image, median_sharpness)| intra_station::AnchorCandidate {
                    path: image.filename.clone(),
                    median_sharpness: *median_sharpness,
                })
                .collect::<Vec<_>>();
            intra_station::select_anchor(&candidates)
        })
        .unwrap_or(0);
    let anchor = scored.remove(anchor_index).0;
    scored.sort_by(|(left_image, left_score), (right_image, right_score)| {
        right_score
            .total_cmp(left_score)
            .then_with(|| left_image.filename.cmp(&right_image.filename))
    });
    let mut ordered = Vec::with_capacity(images.len());
    ordered.push(anchor);
    ordered.extend(scored.into_iter().map(|(image, _)| image));
    if std::env::var_os("RAW_EDITOR_STATION_SHARPNESS_TRACE").is_some() {
        let detail = ordered
            .iter()
            .map(|image| {
                let score = crate::panorama_utils::mosaic::station_member_median_sharpness(image);
                format!("{}={score:.6}", image.filename)
            })
            .collect::<Vec<_>>()
            .join(", ");
        println!("  - Focus station anchor order: {detail}");
    }
    ordered
}

/// Test-only entry point for Property 78. This deliberately delegates to the
/// production default-station wiring above, rather than duplicating the
/// rejection predicate in the property harness.
#[cfg(test)]
pub(crate) fn station_members_from_measured_group_join_evidence(
    images: &[&ImageInfo],
    recorded: &HashMap<String, IntraStationFrameRecord>,
) -> (Vec<StationMember>, Vec<GroupJoinRejection>) {
    station_members_from_frame_records(images, recorded)
}
