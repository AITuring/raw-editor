use super::*;

pub(crate) fn focus_stack_is_shifted_mosaic(
    images: &[&ImageInfo],
    global_homographies: &HashMap<usize, Matrix3<f64>>,
    projection: Projection,
) -> bool {
    let Some(first) = images.first() else {
        return false;
    };
    let Some(reference_homography) = global_homographies.get(&first.id) else {
        return false;
    };
    let Some(reference) = mapped_frame_geometry(first, reference_homography, projection) else {
        return false;
    };
    let reference_width = first.width().max(1) as f64;
    let reference_height = first.height().max(1) as f64;
    // Some exported frames lose the equivalent-focal tag. Use the first tag
    // available in the stack as a comparison anchor so a missing tag on the
    // first frame does not hide a later 35mm/85mm switch.
    let reference_focal = first
        .focal_length_35mm
        .or_else(|| images.iter().find_map(|image| image.focal_length_35mm));
    images.iter().skip(1).any(|image| {
        let Some(homography) = global_homographies.get(&image.id) else {
            return false;
        };
        let Some(current) = mapped_frame_geometry(image, homography, projection) else {
            return false;
        };
        let center_shift = (current.center.x - reference.center.x).abs() > reference_width * 0.08
            || (current.center.y - reference.center.y).abs() > reference_height * 0.08;
        let relative_scale = current.scale / reference.scale.max(1e-8);
        let scale_shift = relative_scale.is_finite() && !(0.90..=1.10).contains(&relative_scale);
        let relative_area = current.area / reference.area.max(1e-8);
        let area_shift = relative_area.is_finite() && !(0.82..=1.22).contains(&relative_area);
        let rotation_shift = wrapped_angle_delta(current.rotation, reference.rotation) > 0.035;
        let focal_shift = reference_focal
            .zip(image.focal_length_35mm)
            .map(|(reference_focal, current_focal)| {
                let ratio = current_focal / reference_focal.max(1e-8);
                ratio.is_finite() && !(0.90..=1.10).contains(&ratio)
            })
            .unwrap_or(false);
        center_shift || scale_shift || area_shift || rotation_shift || focal_shift
    })
}

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
    pub(crate) displacement: Option<super::super::mosaic::NativeResidualDisplacement>,
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
    pub(crate) fn unattributed(image: Rgb32FImage) -> Self {
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
pub(crate) fn sample_focus_plane(
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

/// Normalised Sharpness_Score and coverage of every ownership cell of one
/// projected layer (需求 3.2, 3.7).
///
/// A cell counts as covered when the layer's validity mask is set at the cell
/// centre.  The cell is the *decision* unit; a cell only partly inside the
/// projected trapezoid still gets one owner, and the per-pixel write below
/// never copies a pixel the layer does not actually have, so the Coverage_Mask
/// stays exact at pixel resolution.
pub(crate) fn measure_focus_cells(
    geometry: &OwnershipGridGeometry,
    plan: &CellSamplingPlan,
    image: &Rgb32FImage,
    mask: &GrayImage,
    left: u32,
    top: u32,
) -> (Vec<f64>, Vec<bool>) {
    // 需求 3.2: the candidate and the current composite are comparable only when
    // they are measured at the same positions with the same window, and the only
    // way to guarantee that across two call sites is to insist that both measure
    // with the plan this station's ownership grid defines.  A plan from anywhere
    // else would make defocus indistinguishable from a change of sampling
    // geometry, which is the one confusion 需求 3.2 exists to rule out.
    plan.assert_matches(&geometry.sampling_plan());
    let count = geometry.cell_count();
    let mut sharpness = vec![0.0f64; count];
    let mut covered = vec![false; count];
    let columns = geometry.columns as usize;
    let half = f64::from(geometry.cell_size_px) * 0.5;
    sharpness
        .par_iter_mut()
        .zip(covered.par_iter_mut())
        .enumerate()
        .for_each(|(cell, (sharpness, covered))| {
            let (column, row) = ((cell % columns) as u32, (cell / columns) as u32);
            let (origin_x, origin_y) = geometry.cell_origin(column, row);
            let sample = |x: f64, y: f64| sample_focus_plane(image, mask, left, top, x, y);
            if sample(origin_x + half, origin_y + half).is_none() {
                return;
            }
            *covered = true;
            *sharpness = plan.measure(origin_x, origin_y, sample);
        });
    (sharpness, covered)
}

/// Pixel inconsistency of every ownership cell between a candidate layer and
/// the current composite (需求 3.3).
pub(crate) fn measure_focus_cell_disagreement(
    geometry: &OwnershipGridGeometry,
    candidate: &RenderedFocusLayer,
    base: &Rgb32FImage,
    base_mask: &GrayImage,
) -> Vec<f64> {
    let count = geometry.cell_count();
    let columns = geometry.columns as usize;
    let cell_size = f64::from(geometry.cell_size_px);
    let mut disagreement = vec![0.0f64; count];
    disagreement
        .par_iter_mut()
        .enumerate()
        .for_each(|(cell, disagreement)| {
            let (column, row) = ((cell % columns) as u32, (cell / columns) as u32);
            let (origin_x, origin_y) = geometry.cell_origin(column, row);
            // Both closures receive the same fractional cell offsets and map
            // them into their own plane, which is how 需求 3.3's "same cell"
            // comparison stays a property of the code.
            *disagreement = cell_disagreement(
                |fx, fy| {
                    sample_focus_plane(
                        &candidate.image,
                        &candidate.mask,
                        candidate.left,
                        candidate.top,
                        origin_x + fx * cell_size,
                        origin_y + fy * cell_size,
                    )
                },
                |fx, fy| {
                    sample_focus_plane(
                        base,
                        base_mask,
                        0,
                        0,
                        origin_x + fx * cell_size,
                        origin_y + fy * cell_size,
                    )
                },
            );
        });
    disagreement
}

/// Rasterise an ownership cell decision onto a layer's own pixel grid
/// (需求 3.6).
///
/// The result is a hard mask whose boundaries are exactly ownership cell
/// boundaries: no feather, no alpha ramp, no transition band.  The legacy
/// streaming mosaic keeps its `STREAMING_OWNERSHIP_FEATHER` sub-cell blur; the
/// station layer no longer has one.
pub(crate) fn focus_cell_decision_for_layer(
    geometry: &OwnershipGridGeometry,
    took: &[bool],
    layer: &RenderedFocusLayer,
) -> GrayImage {
    let (width, height) = layer.image.dimensions();
    let mut decision = GrayImage::new(width, height);
    let cell_size = geometry.cell_size_px.max(1);
    let columns = geometry.columns as usize;
    let rows = geometry.rows as usize;
    let width_usize = width as usize;
    decision
        .as_mut()
        .par_chunks_mut(width_usize)
        .enumerate()
        .for_each(|(y, row)| {
            let global_y = layer.top as usize + y;
            let cell_row = global_y / cell_size as usize;
            if cell_row >= rows {
                return;
            }
            let base = cell_row * columns;
            for (x, value) in row.iter_mut().enumerate() {
                let cell_column = (layer.left as usize + x) / cell_size as usize;
                if cell_column < columns && took[base + cell_column] {
                    *value = 255;
                }
            }
        });
    decision
}

/// Expand an ownership cell plane to one value per tile pixel (需求 4.6).
pub(crate) fn focus_cell_plane_to_pixels(
    geometry: &OwnershipGridGeometry,
    cells: &[f32],
    width: u32,
    height: u32,
) -> Vec<f32> {
    let mut values = vec![0.0f32; width as usize * height as usize];
    let cell_size = geometry.cell_size_px.max(1) as usize;
    let columns = geometry.columns as usize;
    let rows = geometry.rows as usize;
    values
        .par_chunks_mut(width as usize)
        .enumerate()
        .for_each(|(y, row)| {
            let cell_row = y / cell_size;
            if cell_row >= rows {
                return;
            }
            let base = cell_row * columns;
            for (x, value) in row.iter_mut().enumerate() {
                let cell_column = x / cell_size;
                if cell_column < columns {
                    *value = cells[base + cell_column];
                }
            }
        });
    values
}

/// Expand the cell-level Textured_Pixel evidence onto the station plane.
pub(crate) fn focus_cell_texture_to_pixels(
    geometry: &OwnershipGridGeometry,
    cells: &[bool],
    width: u32,
    height: u32,
) -> Vec<u8> {
    let mut values = vec![0u8; width as usize * height as usize];
    let cell_size = geometry.cell_size_px.max(1) as usize;
    let columns = geometry.columns as usize;
    let rows = geometry.rows as usize;
    values
        .par_chunks_mut(width as usize)
        .enumerate()
        .for_each(|(y, row)| {
            let cell_row = y / cell_size;
            if cell_row >= rows {
                return;
            }
            let base = cell_row * columns;
            for (x, value) in row.iter_mut().enumerate() {
                let cell_column = x / cell_size;
                if cell_column < columns && cells[base + cell_column] {
                    *value = 255;
                }
            }
        });
    values
}
