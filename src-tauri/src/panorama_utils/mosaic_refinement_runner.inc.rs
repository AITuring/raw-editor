/// Native patch refinement against an in-memory canvas (需求 2.3–2.8).
fn refine_native_layer(
    base: &Rgb32FImage,
    base_mask: &GrayImage,
    sampler: &mut LayerSampler<'_>,
    width: u32,
    height: u32,
    mode: NativeRefineMode,
) -> NativeRefineOutcome {
    let mut outcome = NativeRefineOutcome::default();
    if mode == NativeRefineMode::LegacyMosaic && sampler.scale <= 2.0 {
        return outcome;
    }
    outcome.skipped = false;
    outcome.control_point_spacing_px = native_control_point_spacing_px(sampler.scale);
    let entry_field = sampler.residual.clone();
    let cell_area_px = (NATIVE_REFINE_STEP as f64 * sampler.scale).powi(2);
    let mut accepted_cells = NativeRefinePass::new(width, height).accepted_grid;
    let base = NativeRefineBase {
        image: base,
        mask: base_mask,
        left: 0,
        top: 0,
    };
    for (index, spacing) in [2.0, 1.0].into_iter().enumerate() {
        let mut pass = NativeRefinePass::new(width, height);
        collect_native_observations(&base, sampler, width, height, spacing, mode, &mut pass);
        apply_native_pass(
            pass,
            sampler,
            &mut outcome,
            &mut accepted_cells,
            cell_area_px,
            spacing,
            index == 0,
            mode,
        );
    }
    finalize_native_refinement(&mut outcome, sampler, entry_field, mode);
    outcome
}

/// Native patch refinement against a tiled canvas (需求 2.3–2.8).
///
/// This is the same measurement as [`refine_native_layer`]; only the base pixels
/// arrive one tile at a time, because the streaming compositor is used exactly
/// when the canvas is too large to hold.  Without this entry point the whole of
/// 需求 2.3–2.9 would be unreachable for every multi-station stack, which is the
/// only kind of stack the streaming compositor ever sees.
#[allow(clippy::too_many_arguments)]
fn refine_native_layer_streaming(
    store: &StreamingMosaicStore,
    owner: Option<u8>,
    sampler: &mut LayerSampler<'_>,
    width: u32,
    height: u32,
    layer: (u32, u32, u32, u32),
    mode: NativeRefineMode,
) -> Result<NativeRefineOutcome, String> {
    let mut outcome = NativeRefineOutcome {
        skipped: false,
        control_point_spacing_px: native_control_point_spacing_px(sampler.scale),
        ..Default::default()
    };
    let entry_field = sampler.residual.clone();
    let cell_area_px = (NATIVE_REFINE_STEP as f64 * sampler.scale).powi(2);
    let mut accepted_cells = NativeRefinePass::new(width, height).accepted_grid;
    let (left, right, top, bottom) = layer;
    for (index, spacing) in [2.0, 1.0].into_iter().enumerate() {
        let mut pass = NativeRefinePass::new(width, height);
        for tile_row in top / store.tile_size..=bottom / store.tile_size {
            for tile_column in left / store.tile_size..=right / store.tile_size {
                let (tile_left, tile_top, tile_width, tile_height) =
                    store.tile_extent(tile_column, tile_row);
                let (tile, covered) = store.load_tile(tile_column, tile_row)?;
                // A later focal plane of a station must register against its own
                // camera position: pixels owned by another station are available
                // for colour matching but never for residual motion.
                let tile_mask = match owner {
                    Some(group) => {
                        let tile_owner = store.load_owner_tile(tile_column, tile_row);
                        GrayImage::from_fn(tile_width, tile_height, |x, y| {
                            Luma([
                                if covered.get_pixel(x, y)[0] != 0
                                    && tile_owner.get_pixel(x, y)[0] == group
                                {
                                    255
                                } else {
                                    0
                                },
                            ])
                        })
                    }
                    None => covered,
                };
                if tile_mask.as_raw().iter().all(|&value| value == 0) {
                    continue;
                }
                let base = NativeRefineBase {
                    image: &tile,
                    mask: &tile_mask,
                    left: tile_left,
                    top: tile_top,
                };
                collect_native_observations(
                    &base, sampler, width, height, spacing, mode, &mut pass,
                );
            }
        }
        apply_native_pass(
            pass,
            sampler,
            &mut outcome,
            &mut accepted_cells,
            cell_area_px,
            spacing,
            index == 0,
            mode,
        );
    }
    finalize_native_refinement(&mut outcome, sampler, entry_field, mode);
    Ok(outcome)
}

/// Distance between neighbouring control points in native pixels (需求 2.3).
///
/// [`NATIVE_REFINE_STEP`] is a step in *analysis* units, so the native spacing
/// is that step times the layer's analysis scale.  Published so the default
/// station path can compare the measured spacing against the Focus_Fuser's
/// ownership cell side.
pub(crate) fn native_control_point_spacing_px(analysis_scale: f64) -> f64 {
    NATIVE_REFINE_STEP as f64 * analysis_scale
}

/// The displacement field a native refinement left behind, addressed in canvas
/// pixels (需求 2.3–2.7).
///
/// The refinement measures and corrects in the layer's analysis units; a caller
/// that already holds the layer rendered on the canvas needs the same
/// correction in canvas pixels, which is what [`Self::at`] returns.
#[derive(Debug, Clone)]
pub(crate) struct NativeResidualDisplacement {
    field: Field<2>,
    left: u32,
    top: u32,
    scale: f64,
}

impl NativeResidualDisplacement {
    #[cfg(test)]
    pub(crate) fn from_test_field(field: Field<2>) -> Self {
        Self {
            field,
            left: 0,
            top: 0,
            scale: 1.0,
        }
    }

    /// Canvas-pixel offset to add to a canvas position before sampling the
    /// layer, i.e. exactly what [`LayerSampler::source_point`] adds.
    pub(crate) fn at(&self, canvas_x: f64, canvas_y: f64) -> [f64; 2] {
        let delta = self.field.at(
            (canvas_x - f64::from(self.left)) / self.scale,
            (canvas_y - f64::from(self.top)) / self.scale,
        );
        [delta[0] * self.scale, delta[1] * self.scale]
    }

    /// Largest correction the field carries, in canvas pixels.  Reported so a
    /// run log shows whether the refinement moved anything at all.
    pub(crate) fn max_offset_px(&self) -> f64 {
        self.field
            .values
            .iter()
            .map(|delta| delta[0].hypot(delta[1]) * self.scale)
            .fold(0.0, f64::max)
    }
}

/// One frame of the Intra_Station_Registrar on the default station path
/// (需求 2.2–2.9).
pub(crate) struct StationFrameRegistration {
    /// The `intra_station[*].frames[*]` entry of 需求 2.9.
    pub(crate) record: IntraStationFrameRecord,
    /// The correction to apply to the already rendered layer, or `None` when
    /// nothing was measured.
    pub(crate) displacement: Option<NativeResidualDisplacement>,
    /// Control points the final pass measured, for the run log.
    pub(crate) evaluated_control_points: usize,
}

/// Register one non-anchor frame of a Capture_Station against the station's
/// already composited canvas at native resolution (需求 2.2–2.9).
///
/// This is the entry point of the *default* station path.  It runs the same
/// measurement as the mosaic comparison path — [`refine_layer`] twice at the
/// analysis resolution of 需求 2.2, then [`refine_native_layer`] with the
/// control point gates of 需求 2.3–2.8 — over a [`LayerSampler`] built from the
/// frame's own source pixels, and hands back both the report record and the
/// displacement field, so the caller's already rendered layer can be corrected
/// by exactly what was measured.
///
/// `layer` is the frame's canvas footprint as the caller rendered it, and
/// `transform` must be the pose that produced it: the sampler reproduces that
/// geometry, which is what makes the measured displacement applicable to those
/// pixels.
#[allow(clippy::too_many_arguments)]
pub(crate) fn register_station_frame_native(
    base: &Rgb32FImage,
    base_mask: &GrayImage,
    info: &ImageInfo,
    source: &Rgb32FImage,
    transform: &Matrix3<f64>,
    projection: Projection,
    offset: (f64, f64),
    layer: (u32, u32, u32, u32),
    anchor_long_side: u32,
    into_anchor: Matrix3<f64>,
) -> Option<StationFrameRegistration> {
    let inverse = transform.try_inverse()?;
    let (left, right, top, bottom) = layer;
    if right <= left || bottom <= top {
        return None;
    }
    let layer_width = right - left + 1;
    let layer_height = bottom - top + 1;
    // 需求 2.2: the two global + local rounds run at an analysis resolution whose
    // long side stays inside INTRA_STATION_ANALYSIS_LONG_SIDE; the native patch
    // refinement then runs on the native samples this sampler reaches.
    let scale = (layer_width.max(layer_height) as f64 / ANALYSIS_LONG_SIDE as f64).max(1.0);
    let analysis_width = (layer_width as f64 / scale).ceil() as u32;
    let analysis_height = (layer_height as f64 / scale).ceil() as u32;
    let mut sampler = LayerSampler {
        info,
        source,
        source_divisor: 1.0,
        inverse,
        projection,
        offset,
        left,
        top,
        scale,
        residual: Field::new(analysis_width, analysis_height, GRID_STEP as f64),
    };
    for _ in 0..intra_station::INTRA_STATION_ANALYSIS_ROUNDS {
        refine_layer(
            base,
            base_mask,
            &mut sampler,
            analysis_width,
            analysis_height,
        );
    }
    let outcome = refine_native_layer(
        base,
        base_mask,
        &mut sampler,
        analysis_width,
        analysis_height,
        NativeRefineMode::IntraStation { anchor_long_side },
    );
    let record = intra_station_frame_record(&info.filename, into_anchor, &outcome);
    Some(StationFrameRegistration {
        record,
        displacement: (!outcome.skipped).then(|| NativeResidualDisplacement {
            field: sampler.residual,
            left,
            top,
            scale,
        }),
        evaluated_control_points: outcome.evaluated_control_points,
    })
}

