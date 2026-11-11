//! Build the original-source payload only for consumers that sample owners.
//!
//! Fusion and its source evidence are prepared before this helper. Skipping an
//! unused payload does not change the fused raster, ownership or refinement.

use crate::panorama_utils::shared::ImageInfo;
use crate::panorama_utils::stack_pipeline::report::TilePayloadReport;
use crate::panorama_utils::stitching::{
    FocusSourceRefinement, FocusStackTileMasks, FocusTileLoadPurpose, FocusTileSource,
    FocusTileSources, NativeOwnerGeometry, resolve_native_owner_geometry,
};
use image::Rgb32FImage;
use nalgebra::Point2;
use std::path::Path;

/// One unchanged legend slot. Zero-owned entries may omit metadata, but cannot
/// shift the slots of later owners or cause a fallback to capture-group order.
pub(super) struct OwnerSlot<'a> {
    pub member: Option<&'a ImageInfo>,
    pub refinement: Option<&'a FocusSourceRefinement>,
    pub owned_pixels: u64,
}

pub(super) fn validated_owner_slots<'a>(
    masks: &'a FocusStackTileMasks,
    members: &[&'a ImageInfo],
) -> Result<Vec<OwnerSlot<'a>>, String> {
    // One ownership scan per payload preparation, shared with the load loop.
    let counts = masks.ownership.owned_pixel_counts();
    masks
        .ownership
        .legend()
        .iter()
        .zip(counts)
        .map(|(path, owned_pixels)| {
            let member = members
                .iter()
                .copied()
                .find(|source| Path::new(&source.filename) == path);
            let refinement = member.and_then(|source| masks.source_refinements.get(&source.id));
            if owned_pixels > 0 {
                let source = member
                    .ok_or_else(|| format!("Missing original for owner {}", path.display()))?;
                if members
                    .iter()
                    .filter(|candidate| Path::new(&candidate.filename) == path)
                    .count()
                    != 1
                {
                    return Err(format!("Ambiguous original for owner {}", path.display()));
                }
                let sampling = refinement
                    .and_then(|r| r.source_sampling.as_ref())
                    .ok_or_else(|| format!("Missing source sampling for {}", path.display()))?;
                let (id, filename, dimensions) = sampling.snapshot_metadata();
                if id != source.id || Path::new(filename) != path {
                    return Err(format!(
                        "Source sampling identity mismatch for {}",
                        path.display()
                    ));
                }
                if dimensions != source.dimensions() || dimensions.0 == 0 || dimensions.1 == 0 {
                    return Err(format!(
                        "Source sampling dimensions mismatch for {}: member {:?}, snapshot {:?}",
                        path.display(),
                        source.dimensions(),
                        dimensions
                    ));
                }
            }
            Ok(OwnerSlot {
                member,
                refinement,
                owned_pixels,
            })
        })
        .collect()
}

pub(super) struct OwnerEvidence {
    pub geometry: NativeOwnerGeometry,
    pub raw_owner: u16,
    pub confidence: f32,
    pub textured: u8,
    pub sharpness:
        Option<crate::panorama_utils::stack_pipeline::focus_fuser::SharpnessCellEvidence>,
}

/// The final RAW owner and every cell-derived observation use exactly the same
/// geometric choice as native RGB sampling. Slots and raw IDs are prepared once
/// per station before this per-pixel call; no maps or vectors are built here.
pub(super) fn resolve_owner_evidence(
    masks: &FocusStackTileMasks,
    slots: &[OwnerSlot<'_>],
    raw_owner_ids: &[u16],
    origin: (f64, f64),
    point: Point2<f64>,
) -> Option<OwnerEvidence> {
    let geometry = resolve_native_owner_geometry(point, &masks.ownership, origin, |index| {
        let slot = &slots[index];
        (
            slot.refinement.expect("validated owned refinement"),
            slot.member.expect("validated owned member").dimensions(),
        )
    })?;
    let (x, y) = geometry.evidence_pixel;
    Some(OwnerEvidence {
        raw_owner: raw_owner_ids[usize::from(geometry.local_owner - 1)],
        geometry,
        confidence: masks.confidence.value_at(x, y),
        textured: masks
            .textured
            .get(y as usize * masks.ownership.dimensions().0 as usize + x as usize)
            .copied()
            .unwrap_or(0),
        sharpness: masks.sharpness_evidence.at_pixel(x, y),
    })
}

pub(super) fn record_request(stats: &mut TilePayloadReport, purpose: FocusTileLoadPurpose) {
    match purpose {
        FocusTileLoadPurpose::RasterOnly => stats.raster_only_requests += 1,
        FocusTileLoadPurpose::OwnerSampling => stats.owner_sampling_requests += 1,
    }
}

pub(super) fn load_owner_sources<F>(
    purpose: FocusTileLoadPurpose,
    masks: Option<&FocusStackTileMasks>,
    origin: (f64, f64),
    members: &[&ImageInfo],
    load_image: &mut F,
    stats: &mut TilePayloadReport,
) -> Result<Option<FocusTileSources>, String>
where
    F: FnMut(&ImageInfo) -> Result<Rgb32FImage, String>,
{
    // Return before scanning ownership, resolving paths/refinements or loading
    // any source. Raster consumers still use the caller's complete fusion.
    if purpose == FocusTileLoadPurpose::RasterOnly {
        stats.raster_only_payload_skips += usize::from(masks.is_some());
        return Ok(None);
    }
    let Some(masks) = masks else {
        return Ok(None);
    };
    stats.owner_payload_build_requests += 1;
    let slots = validated_owner_slots(masks, members)?;
    let mut originals = Vec::with_capacity(masks.ownership.legend().len());
    for (path, slot) in masks.ownership.legend().iter().zip(slots) {
        // Excluded members retain their legend slot, but own no pixel and need
        // neither a source sampling map nor an original buffer.
        if slot.owned_pixels == 0 {
            originals.push(None);
            continue;
        }
        let source = slot.member.expect("validated owned member");
        let refinement = slot.refinement.expect("validated owned refinement").clone();
        stats.owner_source_load_requests += 1;
        let image = load_image(source)?;
        if image.dimensions() != source.dimensions() {
            return Err(format!(
                "Prepared source dimensions mismatch for {}: member {:?}, buffer {:?}",
                path.display(),
                source.dimensions(),
                image.dimensions()
            ));
        }
        originals.push(Some(FocusTileSource { image, refinement }));
    }
    Ok(Some(FocusTileSources {
        ownership: masks.ownership.clone(),
        origin,
        originals,
    }))
}

#[cfg(test)]
#[path = "tile_payload_tests.rs"]
mod tests;
