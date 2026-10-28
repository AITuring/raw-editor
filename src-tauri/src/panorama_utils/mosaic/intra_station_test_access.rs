use super::*;

/// Everything one control point measured, plus the verdict the production
/// pass gave it.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ControlPointProbe {
    /// Position in the refinement grid, row major.
    pub(crate) column: usize,
    pub(crate) row: usize,
    /// The cell centre lies on covered base (需求 2.8).
    pub(crate) covered: bool,
    /// The patch pair was complete, so a match was attempted.
    pub(crate) evaluated: bool,
    /// Displacement of a completed bidirectional match, in native pixels of
    /// the anchor frame, before the 需求 2.4 / 2.6 gates.  `None` when the
    /// matcher's similarity gates or the reverse match stopped the
    /// measurement, i.e. when there is no located match to judge.
    pub(crate) displacement_px: Option<[f64; 2]>,
    /// Bidirectional round trip of that match, in native pixels (需求 2.4).
    pub(crate) round_trip_px: f64,
    /// Symmetric reprojection error of the global model there, in native
    /// pixels (需求 2.6).
    pub(crate) symmetric_error_px: f64,
    /// Forward peak correlation, `None` when no hypothesis was scored.
    pub(crate) best_correlation: Option<f64>,
    /// The displacement the production pass accepted at this position, i.e.
    /// what the 需求 2.5 neighbourhood gate let through.
    pub(crate) accepted_px: Option<[f64; 2]>,
}

/// One native refinement pass, measured control point by control point.
///
/// Returns every grid position together with the accepted set of the real
/// pass and the observation positions that entered the displacement field.
pub(crate) fn probe_native_pass(
    base: &Rgb32FImage,
    layer: &Rgb32FImage,
    anchor_long_side: u32,
    spacing: f64,
) -> (Vec<ControlPointProbe>, Vec<(usize, usize)>, usize) {
    let (width, height) = base.dimensions();
    let info = image_info_for_probe(layer);
    let mask = GrayImage::from_pixel(width, height, Luma([255]));
    let sampler = LayerSampler {
        info: &info,
        source: layer,
        source_divisor: 1.0,
        inverse: Matrix3::identity(),
        native_mapping: None,
        projection: Projection::Planar,
        offset: (0.0, 0.0),
        left: 0,
        top: 0,
        scale: 1.0,
        residual: Field::new(width, height, GRID_STEP as f64),
    };
    let mode = NativeRefineMode::IntraStation { anchor_long_side };
    let window = NativeRefineBase {
        image: base,
        mask: &mask,
        left: 0,
        top: 0,
    };
    let mut pass = NativeRefinePass::new(width, height);
    collect_native_observations(&window, &sampler, width, height, spacing, mode, &mut pass);

    let (match_radius, search_samples) = mode.patch_geometry();
    let columns = (20..width.saturating_sub(20))
        .step_by(NATIVE_REFINE_STEP as usize)
        .collect::<Vec<_>>();
    let rows = (20..height.saturating_sub(20))
        .step_by(NATIVE_REFINE_STEP as usize)
        .collect::<Vec<_>>();
    let mut probes = Vec::with_capacity(columns.len() * rows.len());
    for (row, &y) in rows.iter().enumerate() {
        for (column, &x) in columns.iter().enumerate() {
            // `collect_native_observations` took `&sampler`, so the sampler is
            // in the state it measured with and re-measuring one control point
            // reproduces its own measurement exactly.
            let measurement = measure_native_control_point(
                &window,
                &sampler,
                x,
                y,
                spacing,
                mode,
                match_radius,
                search_samples,
            );
            probes.push(ControlPointProbe {
                column,
                row,
                covered: measurement.covered,
                evaluated: measurement.evaluated,
                displacement_px: measurement.measured.map(|(d, _, _, _)| d),
                round_trip_px: measurement.round_trip_px,
                symmetric_error_px: measurement
                    .measured
                    .map(|(_, _, _, error)| error)
                    .unwrap_or(0.0),
                best_correlation: measurement.best_correlation,
                accepted_px: pass.accepted_grid.accepted_at(column, row),
            });
        }
    }
    let observations = pass
        .observations
        .iter()
        .map(|(position, _, _)| {
            let column = columns
                .iter()
                .position(|&x| f64::from(x) == position.x)
                .expect("an observation sits on a grid column");
            let row = rows
                .iter()
                .position(|&y| f64::from(y) == position.y)
                .expect("an observation sits on a grid row");
            (column, row)
        })
        .collect();
    (probes, observations, pass.evaluated)
}

/// What a whole native refinement decided about one frame (需求 2.7 / 2.8).
#[derive(Clone, Copy, Debug)]
pub(crate) struct RefinementProbe {
    /// Accepted control point cells over *both* spacing passes, which is the
    /// numerator 需求 2.8 defines.
    pub(crate) accepted_control_points: usize,
    /// Accepted control point cells of the final spacing pass alone.
    pub(crate) last_pass_accepted_control_points: usize,
    /// Control point cells whose centre lay on covered base in the final
    /// pass: the denominator of 需求 2.8, in cells.
    pub(crate) valid_cells: usize,
    /// Area of one control point cell, in native pixels of the anchor frame.
    pub(crate) cell_area_px: f64,
    pub(crate) evaluated_control_points: usize,
    pub(crate) local_baseline_samples: usize,
    pub(crate) global_baseline_samples: usize,
    /// Median symmetric reprojection error after the refinement, in native
    /// pixels (需求 2.7).
    pub(crate) local_median_symmetric_error_px: f64,
    /// The same median for the global model (需求 2.7).
    pub(crate) global_median_symmetric_error_px: f64,
    pub(crate) local_field_verdict: intra_station::LocalFieldVerdict,
    pub(crate) reverted_to_global: bool,
    pub(crate) inlier_area_coverage: f64,
    pub(crate) status: IntraStationFrameStatus,
    /// `true` when the displacement field the sampler carries afterwards is
    /// bit for bit the one it entered with, i.e. every control point position
    /// kept the global model's displacement.
    pub(crate) field_unchanged: bool,
}

/// Drive the whole native refinement of one frame (both spacing passes).
pub(crate) fn probe_native_refinement(
    base: &Rgb32FImage,
    layer: &Rgb32FImage,
    anchor_long_side: u32,
) -> RefinementProbe {
    let (width, height) = base.dimensions();
    let info = image_info_for_probe(layer);
    let mask = GrayImage::from_pixel(width, height, Luma([255]));
    let entry = Field::new(width, height, GRID_STEP as f64);
    let mut sampler = LayerSampler {
        info: &info,
        source: layer,
        source_divisor: 1.0,
        inverse: Matrix3::identity(),
        native_mapping: None,
        projection: Projection::Planar,
        offset: (0.0, 0.0),
        left: 0,
        top: 0,
        scale: 1.0,
        residual: entry.clone(),
    };
    let outcome = refine_native_layer(
        base,
        &mask,
        &mut sampler,
        width,
        height,
        NativeRefineMode::IntraStation { anchor_long_side },
    );
    let field_unchanged = sampler.residual.values.len() == entry.values.len()
        && sampler
            .residual
            .values
            .iter()
            .zip(entry.values.iter())
            .all(|(left, right)| {
                left[0].to_bits() == right[0].to_bits() && left[1].to_bits() == right[1].to_bits()
            });
    RefinementProbe {
        accepted_control_points: outcome.accepted_control_points,
        last_pass_accepted_control_points: outcome.last_pass_accepted_control_points,
        valid_cells: outcome.valid_cells,
        cell_area_px: outcome.cell_area_px,
        evaluated_control_points: outcome.evaluated_control_points,
        local_baseline_samples: outcome.local_baseline_samples,
        global_baseline_samples: outcome.global_baseline_samples,
        local_median_symmetric_error_px: outcome.local_median_symmetric_error_px,
        global_median_symmetric_error_px: outcome.global_median_symmetric_error_px,
        local_field_verdict: outcome.local_field_verdict,
        reverted_to_global: outcome.reverted_to_global,
        inlier_area_coverage: outcome.inlier_area_coverage,
        status: outcome.frame_status(),
        field_unchanged,
    }
}

fn image_info_for_probe(image: &Rgb32FImage) -> ImageInfo {
    ImageInfo {
        id: 0,
        filename: "/tmp/stack-pipeline/intra-station-probe.NEF".to_string(),
        width: image.width(),
        height: image.height(),
        alignment_image: image::DynamicImage::ImageRgb32F(image.clone()).to_luma8(),
        full_image: None,
        scale_factor: 1.0,
        focal_length_35mm: Some(75.0),
        source_footprint: None,
        overview_reference: false,
        features: Vec::new(),
        top_features: Vec::new(),
        foreground_range: None,
        foreground_mask: None,
        horizontal_edge_rows: Vec::new(),
        vertical_edge_columns: Vec::new(),
    }
}
