//! Shared geometric owner selection for native RGB sampling and RAW evidence.
//!
//! Candidates come only from the compositor's existing 3×3 ownership stencil.
//! Colour, sharpness and interpolation results never participate in selection.

use super::{FocusSourceRefinement, OwnershipMap};
use nalgebra::{Point2, Point3};

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct NativeOwnerGeometry {
    /// One-based identifier in the station's unchanged ownership legend.
    pub(crate) local_owner: u16,
    /// The ownership-plane cell that supplied this candidate and its evidence.
    pub(crate) evidence_pixel: (u32, u32),
    /// Original prepared-source coordinates, without rounding or clamping.
    pub(crate) prepared_point: Point2<f64>,
    pub(crate) used_fallback: bool,
}

#[derive(Clone, Copy)]
struct Candidate {
    owner: u16,
    pixel: (u32, u32),
    distance: f64,
    order: u8,
}

/// Resolve the nearest geometrically supported distinct owner, preserving the
/// original nearest-owner selection whenever it can sample the point.
///
/// The callback addresses the zero-based legend slot. Before calling this in a
/// pixel loop, callers must validate that every owner with pixels has a source,
/// sampling metadata and its exact prepared-buffer dimensions. Missing metadata
/// is a payload error, not grounds to try another owner. The immutable sampling
/// metadata and dimensions must be the same for rendering and evidence replay.
pub(crate) fn resolve_native_owner_geometry<'a>(
    point: Point2<f64>,
    ownership: &OwnershipMap,
    origin: (f64, f64),
    mut source: impl FnMut(usize) -> (&'a FocusSourceRefinement, (u32, u32)),
) -> Option<NativeOwnerGeometry> {
    if !point.x.is_finite() || !point.y.is_finite() {
        return None;
    }
    let (width, height) = ownership.dimensions();
    let rounded = (point.x.round(), point.y.round());
    let mut candidates = [Candidate {
        owner: 0,
        pixel: (0, 0),
        distance: f64::INFINITY,
        order: 0,
    }; 9];
    let mut count = 0;
    for dy in -1..=1 {
        for dx in -1..=1 {
            let x = rounded.0 + f64::from(dx);
            let y = rounded.1 + f64::from(dy);
            if x < 0.0 || y < 0.0 || x >= f64::from(width) || y >= f64::from(height) {
                continue;
            }
            let owner = ownership.owner_at(x as u32, y as u32);
            if owner == 0 {
                continue;
            }
            // Keep the old arithmetic and strict tie rule: no mul_add and no
            // owner-ID ordering. Duplicate owners retain their nearest cell.
            let candidate = Candidate {
                owner,
                pixel: (x as u32, y as u32),
                distance: (x - point.x).powi(2) + (y - point.y).powi(2),
                order: ((dy + 1) * 3 + dx + 1) as u8,
            };
            if let Some(existing) = candidates[..count].iter_mut().find(|c| c.owner == owner) {
                if candidate.distance < existing.distance {
                    *existing = candidate;
                }
            } else {
                candidates[count] = candidate;
                count += 1;
            }
        }
    }
    // At most nine stack-resident candidates; sorting allocates no heap. The
    // stored stencil order preserves row-major ties even after deduplication.
    candidates[..count].sort_unstable_by(|a, b| {
        a.distance
            .partial_cmp(&b.distance)
            .expect("finite ownership-stencil distance")
            .then(a.order.cmp(&b.order))
    });
    for (attempt, candidate) in candidates[..count].iter().enumerate() {
        let (refinement, dimensions) = source(usize::from(candidate.owner - 1));
        let corrected = refinement.inverse_canvas_point(point);
        let Some(prepared_point) = refinement
            .source_sampling
            .as_ref()
            .expect("validated native owner must have source sampling metadata")
            .source_point(Point3::new(
                corrected.x + origin.0,
                corrected.y + origin.1,
                1.0,
            ))
        else {
            continue;
        };
        // source_point checks the calibrated source footprint and metadata
        // bounds. Preserve the additional actual-buffer bounds check too.
        if prepared_point.x < 0.0
            || prepared_point.y < 0.0
            || prepared_point.x >= f64::from(dimensions.0)
            || prepared_point.y >= f64::from(dimensions.1)
        {
            continue;
        }
        return Some(NativeOwnerGeometry {
            local_owner: candidate.owner,
            evidence_pixel: candidate.pixel,
            prepared_point,
            used_fallback: attempt != 0,
        });
    }
    None
}

#[cfg(test)]
#[path = "native_owner_geometry_tests.rs"]
mod tests;
