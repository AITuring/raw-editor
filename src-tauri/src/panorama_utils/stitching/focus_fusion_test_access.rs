use super::*;

/// Per-cell normalised Sharpness_Score and coverage of one projected layer,
/// exactly as the composite and every candidate of a Capture_Station are
/// measured.
pub(crate) fn measure_cells(
    geometry: &OwnershipGridGeometry,
    plan: &CellSamplingPlan,
    image: &Rgb32FImage,
    mask: &GrayImage,
    left: u32,
    top: u32,
) -> (Vec<f64>, Vec<bool>) {
    measure_focus_cells(geometry, plan, image, mask, left, top)
}
