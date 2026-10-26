/// A source entry used by the runtime adapter. `owner` is the value copied to
/// the final ownership plane; IDs need not be contiguous.
#[derive(Debug, Clone)]
pub(crate) struct QualitySource {
    pub owner: u16,
    pub path: String,
    pub geometry: SourceGeometry,
    pub dimensions: (u32, u32),
    pub native_registration: Option<NativeReferenceMap>,
}

/// Replays the actual station-canvas inverse refinement before the RAW inverse.
#[derive(Debug, Clone)]
pub(crate) struct NativeReferenceMap {
    pub world_to_station: Matrix3<f64>,
    pub station_to_world: Matrix3<f64>,
    pub refinement: crate::panorama_utils::stitching::FocusSourceRefinement,
    pub station_origin: (f64, f64),
}

impl NativeReferenceMap {
    fn corrected_station(&self, world: Point2<f64>) -> Option<Point2<f64>> {
        let station = project(&self.world_to_station, world)?;
        Some(self.refinement.inverse_canvas_point(station))
    }

    fn reference_world(&self, world: Point2<f64>) -> Option<Point2<f64>> {
        project(&self.station_to_world, self.corrected_station(world)?)
    }
}

/// Inputs retained by the compositor after the final image is assembled.
/// `textured` carries the cell-level Textured_Pixel evidence. An absent plane
/// is reported as insufficient evidence instead of widening the statistic to
/// all covered pixels.
pub(crate) struct QualityGateInput<'a> {
    pub output: &'a Rgb32FImage,
    pub coverage: &'a GrayImage,
    pub ownership: &'a [u16],
    pub confidence: &'a [f32],
    pub textured: Option<&'a [u8]>,
    /// Quantised owner shortfall in hundredths (0..=100) for each output
    /// pixel. Keeping this as bytes avoids another full-size f32 plane.
    pub owner_shortfall: Option<&'a [u8]>,
    /// Quantised ownership disagreement in hundredths (0..=100).
    pub owner_disagreement: Option<&'a [u8]>,
    /// Covered output pixels for which the source reverse lookup could not
    /// establish an owner. These are technical evidence gaps.
    pub unresolved_pixel_count: u64,
    pub world_origin: (f64, f64),
    pub sources: &'a [QualitySource],
    /// Accepted Station_Relations as normalized station-index pairs. Boundary
    /// evidence outside these real overlaps is not a production seam metric.
    pub accepted_station_relations: &'a [(usize, usize)],
    pub residual: &'a ResidualWarp,
    pub acceptance_render_scale: f64,
    pub final_sharpen_amount: f64,
}

pub(crate) type SourceLoader<'a> = dyn FnMut(&QualitySource) -> Result<Rgb32FImage, String> + 'a;

pub(crate) struct Criterion {
    name: &'static str,
    threshold: f64,
    min: Option<f64>,
    max: Option<f64>,
    measured: Vec<FailedMeasurementRecord>,
    failed: Vec<FailedMeasurementRecord>,
    unmeasurable: usize,
    content_not_applicable: usize,
    technical_unmeasurable: usize,
    unmeasurable_reasons: BTreeMap<String, usize>,
    pub(super) diagnostic: bool,
    observed_count: usize,
    /// Keep reports bounded when a large canvas has many boundary samples.
    /// The count remains exact while only a deterministic prefix is retained
    /// for the per-point diagnostics.
    record_limit: usize,
    textured_pixel_count: Option<u64>,
    excluded_flat_pixel_count: Option<u64>,
    textured_low_confidence_ratio: Option<f64>,
    flat_low_confidence_ratio: Option<f64>,
    textured_shortfall_ratio: Option<f64>,
    flat_shortfall_ratio: Option<f64>,
    shortfall_histogram: Vec<u64>,
    disagreement_veto_count: u64,
    unresolved_pixel_count: u64,
}

#[derive(Debug, Clone, Copy)]
struct ConfidenceCoverageStats {
    textured_pixels: u64,
    flat_pixels: u64,
    textured_low: u64,
    flat_low: u64,
    textured_unknown: u64,
    flat_unknown: u64,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct OwnerSharpnessStats {
    pub textured_pixels: u64,
    pub flat_pixels: u64,
    pub textured_shortfall: u64,
    pub flat_shortfall: u64,
    pub shortfall_histogram: Vec<u64>,
    pub disagreement_veto_count: u64,
    pub unresolved_pixel_count: u64,
}

/// Marker of a covered output pixel without owner evidence in the quantised
/// owner planes.
pub(crate) const OWNER_EVIDENCE_UNKNOWN: u8 = u8::MAX;

/// Quantise a `[0, 1]` owner-evidence value to hundredths.  Rounding up keeps
/// every value strictly above a hundredth-aligned limit strictly above the
/// quantised limit, so `quantize_hundredths(v) > hundredths(limit)` is exactly
/// `v > limit`.  A non-finite value is evidence the lookup could not use.
pub(crate) fn quantize_hundredths(value: f32) -> u8 {
    if !value.is_finite() {
        return OWNER_EVIDENCE_UNKNOWN;
    }
    (value * 100.0).ceil().clamp(0.0, 100.0) as u8
}

/// A hundredth-aligned threshold in the units of [`quantize_hundredths`].
fn hundredths(limit: f64) -> u8 {
    (limit * 100.0).round().clamp(0.0, 100.0) as u8
}

/// Owner shortfall limit of 需求 11.12 in quantised units.
pub(crate) fn owner_shortfall_limit() -> u8 {
    hundredths(f64::from(quality_gate::OWNER_SHARPNESS_SHORTFALL_MAX))
}

/// Disagreement above which the Focus_Fuser's mismatch veto may have kept a
/// less sharp owner, in quantised units.
pub(crate) fn owner_disagreement_veto_limit() -> u8 {
    hundredths(super::focus_fuser::OWNERSHIP_DISAGREEMENT_VETO)
}

/// Runtime owner-sharpness statistic used by the Quality_Gate and P71. The
/// quantised planes are produced during the output->station reverse lookup;
/// this function only counts them and never infers ownership.  The shortfall
/// histogram and the veto count describe the Textured_Pixel that exceed the
/// limit, which is the population the criterion judges.
pub(crate) fn owner_sharpness_stats(
    coverage: &[u8],
    textured: &[u8],
    shortfall: &[u8],
    disagreement: &[u8],
    unresolved_pixel_count: u64,
) -> Result<OwnerSharpnessStats, &'static str> {
    if coverage.len() != textured.len()
        || coverage.len() != shortfall.len()
        || coverage.len() != disagreement.len()
    {
        return Err(degradation::DIAGNOSTICS_ROI_INVALID);
    }
    let shortfall_limit = owner_shortfall_limit();
    let veto_limit = owner_disagreement_veto_limit();
    let mut stats = OwnerSharpnessStats {
        shortfall_histogram: vec![0; 20],
        unresolved_pixel_count,
        ..OwnerSharpnessStats::default()
    };
    for (((&covered, &is_textured), &shortfall), &disagreement) in coverage
        .iter()
        .zip(textured)
        .zip(shortfall)
        .zip(disagreement)
    {
        if covered == 0 || shortfall == OWNER_EVIDENCE_UNKNOWN {
            continue;
        }
        let shortfall_exceeds = shortfall > shortfall_limit;
        if is_textured == 0 {
            stats.flat_pixels += 1;
            stats.flat_shortfall += u64::from(shortfall_exceeds);
            continue;
        }
        stats.textured_pixels += 1;
        if shortfall_exceeds {
            stats.textured_shortfall += 1;
            let bucket = usize::from(shortfall.min(100)) * 20 / 101;
            stats.shortfall_histogram[bucket.min(19)] += 1;
            if disagreement != OWNER_EVIDENCE_UNKNOWN && disagreement > veto_limit {
                stats.disagreement_veto_count += 1;
            }
        }
    }
    Ok(stats)
}

impl Default for Criterion {
    fn default() -> Self {
        Self {
            name: "",
            threshold: 0.0,
            min: None,
            max: None,
            measured: Vec::new(),
            failed: Vec::new(),
            unmeasurable: 0,
            content_not_applicable: 0,
            technical_unmeasurable: 0,
            unmeasurable_reasons: BTreeMap::new(),
            diagnostic: false,
            observed_count: 0,
            record_limit: usize::MAX,
            textured_pixel_count: None,
            excluded_flat_pixel_count: None,
            textured_low_confidence_ratio: None,
            flat_low_confidence_ratio: None,
            textured_shortfall_ratio: None,
            flat_shortfall_ratio: None,
            shortfall_histogram: Vec::new(),
            disagreement_veto_count: 0,
            unresolved_pixel_count: 0,
        }
    }
}

impl Criterion {
    pub(super) fn new(name: &'static str, threshold: f64) -> Self {
        Self {
            name,
            threshold,
            ..Self::default()
        }
    }

    fn bounded(name: &'static str, min: f64, max: f64) -> Self {
        Self {
            name,
            threshold: min,
            min: Some(min),
            max: Some(max),
            ..Self::default()
        }
    }

    fn p95_with_max(name: &'static str, p95: f64, max: f64) -> Self {
        Self {
            name,
            threshold: p95,
            max: Some(max),
            ..Self::default()
        }
    }

    pub(super) fn push(&mut self, world: (f64, f64), value: f64, owner_path: &str, pass: bool) {
        self.observed_count = self.observed_count.saturating_add(1);
        let item = FailedMeasurementRecord {
            world: WorldPoint {
                x: world.0,
                y: world.1,
            },
            measured: value,
            owner_path: owner_path.to_string(),
        };
        if self.measured.len() < self.record_limit {
            self.measured.push(item.clone());
        }
        if !pass && self.failed.len() < self.record_limit {
            self.failed.push(item);
        }
    }

    fn miss(&mut self, reason: &'static str) {
        self.miss_count(reason, 1);
    }

    fn miss_count(&mut self, reason: &'static str, count: u64) {
        let count = usize::try_from(count).unwrap_or(usize::MAX);
        self.unmeasurable = self.unmeasurable.saturating_add(count);
        match classify_unmeasurable_reason(reason) {
            UnmeasurableCategory::ContentNotApplicable => {
                self.content_not_applicable = self.content_not_applicable.saturating_add(count)
            }
            UnmeasurableCategory::Technical => {
                self.technical_unmeasurable = self.technical_unmeasurable.saturating_add(count)
            }
        }
        *self
            .unmeasurable_reasons
            .entry(reason.to_string())
            .or_default() += count;
    }

    pub(super) fn finish(
        self,
        _unmeasurable: &mut Vec<UnmeasurableRecord>,
    ) -> QualityGateCriterionRecord {
        // `unmeasurable` records are appended by the caller while processing
        // each ROI; this method only carries the aggregate count.
        let verdict = criterion_verdict(
            self.name,
            self.observed_count,
            self.technical_unmeasurable,
            !self.failed.is_empty(),
            self.diagnostic,
            self.observed_count == 0
                || (self.name == "owner_sharpness_coverage" && self.unresolved_pixel_count > 0),
        );
        QualityGateCriterionRecord {
            name: self.name.to_string(),
            threshold: self.threshold,
            threshold_min: self.min,
            threshold_max: self.max,
            measurable_count: self.observed_count,
            unmeasurable_count: self.unmeasurable,
            content_not_applicable_count: self.content_not_applicable,
            technical_unmeasurable_count: self.technical_unmeasurable,
            verdict,
            station_statistics: Vec::new(),
            unmeasurable_reasons: self.unmeasurable_reasons,
            measured: self.measured,
            diagnostic: self.diagnostic,
            failed: self.failed,
            textured_pixel_count: self.textured_pixel_count,
            excluded_flat_pixel_count: self.excluded_flat_pixel_count,
            textured_low_confidence_ratio: self.textured_low_confidence_ratio,
            flat_low_confidence_ratio: self.flat_low_confidence_ratio,
            textured_shortfall_ratio: self.textured_shortfall_ratio,
            flat_shortfall_ratio: self.flat_shortfall_ratio,
            shortfall_histogram: self.shortfall_histogram,
            disagreement_veto_count: self.disagreement_veto_count,
            unresolved_pixel_count: self.unresolved_pixel_count,
        }
    }
}

pub(crate) fn criterion_verdict(
    name: &str,
    measurable: usize,
    technical: usize,
    has_failed: bool,
    diagnostic: bool,
    statistics_base_empty: bool,
) -> QualityGateVerdict {
    // Diagnostic criteria describe an unavailable measurement path. They do
    // not establish a pass or fail result, even when a fallback value happens
    // to satisfy the nominal threshold.
    if diagnostic {
        return QualityGateVerdict::NotApplicable;
    }
    let technical_ratio = if measurable + technical == 0 {
        0.0
    } else {
        technical as f64 / (measurable + technical) as f64
    };
    let sparse = matches!(
        name,
        "mtf50_normalized" | "noise_sigma_ratio" | "boundary_stroke_alignment"
    );
    let whole_image = matches!(name, "effective_pixel_count" | "owner_sharpness_coverage");
    let dense = matches!(
        name,
        "local_scale_median"
            | "local_scale_pixel_ratio"
            | "gradient_energy_normalized"
            | "roi_delta_e00"
    );
    if (whole_image && statistics_base_empty)
        || (!whole_image && technical_ratio > quality_gate::QUALITY_MAX_UNMEASURABLE_RATIO)
    {
        QualityGateVerdict::InsufficientEvidence
    } else if has_failed {
        QualityGateVerdict::Fail
    } else if sparse && measurable < quality_gate::QUALITY_MIN_MEASURABLE {
        QualityGateVerdict::NotApplicable
    } else if dense && measurable < quality_gate::QUALITY_MIN_MEASURABLE {
        QualityGateVerdict::InsufficientEvidence
    } else {
        QualityGateVerdict::Pass
    }
}

pub(crate) fn overall_verdict(criteria: &[QualityGateCriterionRecord]) -> QualityGateVerdict {
    if criteria
        .iter()
        .any(|c| c.verdict == QualityGateVerdict::Fail)
    {
        QualityGateVerdict::Fail
    } else if criteria.is_empty()
        || criteria.iter().any(|c| {
            !matches!(
                c.verdict,
                QualityGateVerdict::Pass | QualityGateVerdict::NotApplicable
            )
        })
    {
        QualityGateVerdict::InsufficientEvidence
    } else {
        QualityGateVerdict::Pass
    }
}

fn world(roi: &QualityRoi) -> (f64, f64) {
    roi.world_origin
}

fn output_roi(image: &Rgb32FImage, roi: &QualityRoi) -> Option<Rgb32FImage> {
    let side = roi.side;
    if roi.x.checked_add(side)? > image.width() || roi.y.checked_add(side)? > image.height() {
        return None;
    }
    Some(ImageBuffer::from_fn(side, side, |x, y| {
        *image.get_pixel(roi.x + x, roi.y + y)
    }))
}

fn project(matrix: &Matrix3<f64>, p: Point2<f64>) -> Option<Point2<f64>> {
    let v = matrix * Vector3::new(p.x, p.y, 1.0);
    if !v.iter().all(|x| x.is_finite()) || v.z.abs() < 1e-12 {
        return None;
    }
    Some(Point2::new(v.x / v.z, v.y / v.z))
}

fn source_quad(source: &QualitySource) -> Option<[Point2<f64>; 4]> {
    let matrix = source.geometry.tile_to_world * source.geometry.member_to_anchor;
    let (w, h) = source.dimensions;
    let corners = [
        Point2::new(0.0, 0.0),
        Point2::new(f64::from(w.saturating_sub(1)), 0.0),
        Point2::new(
            f64::from(w.saturating_sub(1)),
            f64::from(h.saturating_sub(1)),
        ),
        Point2::new(0.0, f64::from(h.saturating_sub(1))),
    ];
    Some([
        project(&matrix, corners[0])?,
        project(&matrix, corners[1])?,
        project(&matrix, corners[2])?,
        project(&matrix, corners[3])?,
    ])
}

/// Source sampling through the complete output-to-source mapping, using the
/// same Catmull-Rom kernel as rendering so reference noise is not artificially
/// reduced by a different bilinear filter.
/// This is the single allowed reference resample; output pixels are copied as
/// is by [`output_roi`].
fn source_roi(
    source: &Rgb32FImage,
    roi: &QualityRoi,
    geometry: &SourceGeometry,
    residual: &ResidualWarp,
    native_registration: Option<&NativeReferenceMap>,
) -> Result<Rgb32FImage, &'static str> {
    let inverse = (geometry.tile_to_world * geometry.member_to_anchor)
        .try_inverse()
        .ok_or(degradation::OWNER_SOURCE_UNDECODABLE)?;
    let mut result = Rgb32FImage::new(roi.side, roi.side);
    for y in 0..roi.side {
        for x in 0..roi.side {
            let world = Point2::new(
                roi.world_origin.0 + f64::from(x),
                roi.world_origin.1 + f64::from(y),
            );
            let mapped = if let Some(native) = native_registration {
                let unwarped = residual.warp_inverse(world, geometry.station_id);
                if let Some(sampling) = native.refinement.source_sampling.as_ref() {
                    let station = native
                        .corrected_station(unwarped)
                        .ok_or(degradation::OWNER_SOURCE_UNDECODABLE)?;
                    sampling
                        .source_point(nalgebra::Point3::new(
                            station.x + native.station_origin.0,
                            station.y + native.station_origin.1,
                            1.0,
                        ))
                        .ok_or(degradation::OWNER_SOURCE_UNDECODABLE)?
                } else {
                    let corrected = native
                        .reference_world(unwarped)
                        .ok_or(degradation::OWNER_SOURCE_UNDECODABLE)?;
                    project(&inverse, corrected).ok_or(degradation::OWNER_SOURCE_UNDECODABLE)?
                }
            } else {
                quality_gate::map_world_to_source_with_inverse(
                    world,
                    geometry.station_id,
                    residual,
                    &inverse,
                )?
            };
            let max_x = f64::from(source.width().saturating_sub(1));
            let max_y = f64::from(source.height().saturating_sub(1));
            if !mapped.x.is_finite()
                || !mapped.y.is_finite()
                || mapped.x < 0.0
                || mapped.y < 0.0
                || mapped.x > max_x
                || mapped.y > max_y
            {
                return Err(degradation::OWNER_SOURCE_UNDECODABLE);
            }
            let pixel = crate::panorama_utils::stitching::get_high_quality_interpolated_pixel(
                source, mapped.x, mapped.y,
            );
            result.put_pixel(x, y, pixel);
        }
    }
    Ok(result)
}

pub(super) fn record_unmeasurable(
    criterion: &mut Criterion,
    records: &mut Vec<UnmeasurableRecord>,
    roi: &QualityRoi,
    reason: &'static str,
) {
    let category = classify_unmeasurable_reason(reason);
    criterion.miss(reason);
    records.push(UnmeasurableRecord {
        criterion: criterion.name.to_string(),
        world: WorldPoint {
            x: roi.world_origin.0,
            y: roi.world_origin.1,
        },
        reason: reason.to_string(),
        category,
    });
}

pub(crate) fn classify_unmeasurable_reason(reason: &str) -> UnmeasurableCategory {
    match reason {
        degradation::ROI_NOT_SLANTED_EDGE
        | degradation::SLANTED_EDGE_ANGLE_OUT_OF_RANGE
        | degradation::SLANTED_EDGE_CONTRAST_INSUFFICIENT
        | degradation::SLANTED_EDGE_LINE_FIT_RESIDUAL_EXCEEDED
        | degradation::SLANTED_EDGE_TOO_SHORT
        | degradation::ROI_NOT_FLAT
        | degradation::BOUNDARY_NO_PAIRABLE_EDGE
        | degradation::BOUNDARY_LOW_CONTRAST
        | degradation::BOUNDARY_STROKE_NOT_UNIQUE
        | degradation::BOUNDARY_ORIENTATION_MISMATCH => UnmeasurableCategory::ContentNotApplicable,
        _ => UnmeasurableCategory::Technical,
    }
}

fn ratio(measured: f64, reference: f64, min: f64, max: f64) -> Option<(f64, bool)> {
    quality_gate::compare_ratio(measured, reference, min, max, false).map(|v| (v.value, v.passed))
}

fn gradient_energy_ratio(
    output: &Rgb32FImage,
    reference: &Rgb32FImage,
    local_scale: f64,
) -> Result<Option<(f64, bool)>, &'static str> {
    let output_energy = quality_gate::normalized_gradient_energy(output, local_scale)?;
    // `source_roi` has already resampled Source_RAW onto the output grid, so
    // both sides share the same measured local scale.
    let reference_energy = quality_gate::normalized_gradient_energy(reference, local_scale)?;
    Ok(ratio(
        output_energy,
        reference_energy,
        quality_gate::QUALITY_GRADIENT_RATIO_MIN,
        f64::INFINITY,
    ))
}

/// Keep the observation-only gate bounded on the full-resolution acceptance
/// renders. The exact region labelling and distance transform are useful for
/// ordinary unit-sized fixtures, but a 9k by 11k canvas would allocate and
/// scan several gigabytes just to choose 512px ROIs. A deterministic lattice
/// still checks ownership and coverage over every pixel in a candidate ROI;
/// effective coverage and boundary tracing use separate streaming passes.
fn select_sparse_rois(
    coverage: &GrayImage,
    ownership: &[u16],
    origin: (f64, f64),
) -> Vec<QualityRoi> {
    let (width, height) = coverage.dimensions();
    let side = quality_gate::QUALITY_ROI_SIDE;
    if width < side || height < side {
        return Vec::new();
    }
    let mut result = Vec::new();
    for y in (0..=height - side).step_by(side as usize / 2) {
        for x in (0..=width - side).step_by(side as usize / 2) {
            let owner = ownership[(y * width + x) as usize];
            if owner == 0 || coverage.get_pixel(x, y)[0] == 0 {
                continue;
            }
            let mut same_owner = true;
            for ry in 0..side {
                for rx in 0..side {
                    let index = ((y + ry) * width + x + rx) as usize;
                    if coverage.as_raw()[index] == 0 || ownership[index] != owner {
                        same_owner = false;
                        break;
                    }
                }
                if !same_owner {
                    break;
                }
            }
            if same_owner {
                // A candidate must have a 16px owner/coverage margin on all
                // four sides. Checking this narrow ring is equivalent to the
                // distance-transform predicate used by the exact small-plane
                // selector, while keeping the large-plane path bounded.
                let margin = quality_gate::QUALITY_ROI_MARGIN as u32;
                let margin_ok = x >= margin
                    && y >= margin
                    && x + side + margin <= width
                    && y + side + margin <= height
                    && (y - margin..y + side + margin).all(|row| {
                        (x - margin..x).all(|column| {
                            let index = (row * width + column) as usize;
                            coverage.as_raw()[index] != 0 && ownership[index] == owner
                        }) && (x + side..x + side + margin).all(|column| {
                            let index = (row * width + column) as usize;
                            coverage.as_raw()[index] != 0 && ownership[index] == owner
                        })
                    })
                    && (y - margin..y).all(|row| {
                        (x..x + side).all(|column| {
                            let index = (row * width + column) as usize;
                            coverage.as_raw()[index] != 0 && ownership[index] == owner
                        })
                    })
                    && (y + side..y + side + margin).all(|row| {
                        (x..x + side).all(|column| {
                            let index = (row * width + column) as usize;
                            coverage.as_raw()[index] != 0 && ownership[index] == owner
                        })
                    });
                if !margin_ok {
                    continue;
                }
                result.push(QualityRoi {
                    x,
                    y,
                    side,
                    region_label: result.len() as u32 + 1,
                    owner,
                    world_origin: (origin.0 + f64::from(x), origin.1 + f64::from(y)),
                    boundary_clearance: f64::from(side),
                });
                if result.len() == quality_gate::QUALITY_ROI_MAX_COUNT {
                    return result;
                }
            }
        }
    }
    result
}

fn confidence_coverage_stats(
    coverage: &GrayImage,
    confidence: &[f32],
    textured: &[u8],
) -> Option<ConfidenceCoverageStats> {
    let (width, height) = coverage.dimensions();
    if textured.len() != confidence.len() || confidence.len() != coverage.as_raw().len() {
        return None;
    }
    let mut stats = ConfidenceCoverageStats {
        textured_pixels: 0,
        flat_pixels: 0,
        textured_low: 0,
        flat_low: 0,
        textured_unknown: 0,
        flat_unknown: 0,
    };
    for y in 0..height {
        for x in 0..width {
            let index = (y * width + x) as usize;
            if coverage.as_raw()[index] == 0 {
                continue;
            }
            let is_textured = textured[index] != 0;
            if is_textured {
                stats.textured_pixels = stats.textured_pixels.saturating_add(1);
            } else {
                stats.flat_pixels = stats.flat_pixels.saturating_add(1);
            }
            if !confidence[index].is_finite() {
                if is_textured {
                    stats.textured_unknown = stats.textured_unknown.saturating_add(1);
                } else {
                    stats.flat_unknown = stats.flat_unknown.saturating_add(1);
                }
            } else if confidence[index] < 0.05 {
                if is_textured {
                    stats.textured_low = stats.textured_low.saturating_add(1);
                } else {
                    stats.flat_low = stats.flat_low.saturating_add(1);
                }
            }
        }
    }
    Some(stats)
}

/// Accumulate one output pixel while the compositor maps it back to its
/// ownership cell. Row-major calls preserve the diagnostic sum order without
/// retaining three additional full-resolution score/candidate planes.
pub(crate) fn accumulate_confidence_score(
    result: &mut super::report::ConfidenceScoreDistribution,
    covered: u8,
    confidence: f32,
    textured: u8,
    evidence: super::focus_fuser::SharpnessCellEvidence,
) {
    if covered == 0 || textured == 0 || !confidence.is_finite() || confidence >= 0.05 {
        return;
    }
    result.textured_low_count += 1;
    result.winner_sum += f64::from(evidence.winner_score);
    result.runner_up_sum += f64::from(evidence.runner_up_score);
    let wb = (f64::from(evidence.winner_score).clamp(0.0, 1.0) * 20.0).floor() as usize;
    let rb = (f64::from(evidence.runner_up_score).clamp(0.0, 1.0) * 20.0).floor() as usize;
    result.winner_histogram.resize(21, 0);
    result.runner_up_histogram.resize(21, 0);
    result.winner_histogram[wb.min(20)] += 1;
    result.runner_up_histogram[rb.min(20)] += 1;
    if evidence.winner_score < 0.10 {
        result.winner_below_0_10 += 1;
    } else if evidence.winner_score >= 0.10 && evidence.runner_up_score >= 0.10 {
        result.both_scores_at_least_0_10 += 1;
    } else if evidence.candidate_count <= 1 {
        result.single_candidate_count += 1;
    }
}

/// Count the unique projected source coverage one scanline at a time. This is
/// equivalent to `quality_gate::effective_pixel_coverage`, but never allocates
/// a canvas-sized union bitmap (which is prohibitive for the acceptance
/// renders). Only the current row's sorted intervals are resident.
fn effective_pixel_coverage_streaming(
    coverage: &GrayImage,
    source_quadrilaterals: &[[Point2<f64>; 4]],
    canvas_world_origin: (f64, f64),
) -> Result<EffectivePixelCoverage, &'static str> {
    let (width, height) = coverage.dimensions();
    if width == 0 || height == 0 || source_quadrilaterals.is_empty() {
        return Err(degradation::QUALITY_GATE_INSUFFICIENT_EVIDENCE);
    }
    let output_nontransparent = coverage
        .as_raw()
        .iter()
        .filter(|&&value| value != 0)
        .count() as u64;
    let mut projected_union = 0u64;
    for y in 0..height {
        let scan_y = canvas_world_origin.1 + f64::from(y) + 0.5;
        let mut intervals = Vec::<(f64, f64)>::new();
        for quad in source_quadrilaterals {
            if !quad
                .iter()
                .all(|point| point.x.is_finite() && point.y.is_finite())
            {
                return Err(degradation::DIAGNOSTICS_ROI_INVALID);
            }
            let mut intersections = [0.0f64; 4];
            let mut count = 0usize;
            for edge in 0..4 {
                let a = quad[edge];
                let b = quad[(edge + 1) % 4];
                if ((a.y <= scan_y && scan_y < b.y) || (b.y <= scan_y && scan_y < a.y))
                    && count < intersections.len()
                {
                    intersections[count] = a.x + (scan_y - a.y) * (b.x - a.x) / (b.y - a.y);
                    count += 1;
                }
            }
            if count >= 2 {
                intersections[..count].sort_by(f64::total_cmp);
                for pair in intersections[..count].chunks_exact(2) {
                    let start = (pair[0] - canvas_world_origin.0 - 0.5)
                        .ceil()
                        .max(0.0)
                        .min(f64::from(width)) as u32;
                    let end = (pair[1] - canvas_world_origin.0 - 0.5)
                        .ceil()
                        .max(0.0)
                        .min(f64::from(width)) as u32;
                    if end > start {
                        intervals.push((f64::from(start), f64::from(end)));
                    }
                }
            }
        }
        intervals.sort_by(|left, right| {
            left.0
                .total_cmp(&right.0)
                .then_with(|| left.1.total_cmp(&right.1))
        });
        let mut union = 0u64;
        let mut active: Option<(f64, f64)> = None;
        for (start, end) in intervals {
            let Some((active_start, active_end)) = active else {
                active = Some((start, end));
                continue;
            };
            if start > active_end {
                union = union.saturating_add((active_end - active_start).max(0.0) as u64);
                active = Some((start, end));
            } else if end > active_end {
                active = Some((active_start, end));
            }
        }
        if let Some((active_start, active_end)) = active {
            union = union.saturating_add((active_end - active_start).max(0.0) as u64);
        }
        projected_union = projected_union.saturating_add(union);
    }
    if projected_union == 0 {
        return Err(degradation::QUALITY_GATE_INSUFFICIENT_EVIDENCE);
    }
    Ok(EffectivePixelCoverage {
        output_nontransparent,
        projected_union,
        ratio: output_nontransparent as f64 / projected_union as f64,
    })
}

#[derive(Debug, Clone)]
struct BoundaryEdge {
    offset: f64,
    strength: f64,
    gradient: (f64, f64),
    pixel: (u32, u32),
}

#[derive(Debug, Clone)]
struct BoundaryPairEvidence {
    world: (f64, f64),
    error: f64,
    strength: f64,
    owner_path: String,
}

#[derive(Debug, Clone)]
enum BoundaryCellObservation {
    Measured(BoundaryPairEvidence),
    Unmeasurable {
        world: (f64, f64),
        reason: &'static str,
    },
}

fn boundary_edge_on_side(
    image: &Rgb32FImage,
    coverage: &GrayImage,
    ownership: &[u16],
    seam: (f64, f64),
    normal: (f64, f64),
    side: f64,
    distance: f64,
    expected_owner: u16,
) -> Result<BoundaryEdge, &'static str> {
    const FIXED_CROSS_SECTION_RADIUS_PX: i32 = 2;
    let (width, height) = image.dimensions();
    let tangent = (-normal.1, normal.0);
    let luminance = |pixel: Rgb<f32>| {
        0.2126 * f64::from(pixel[0]) + 0.7152 * f64::from(pixel[1]) + 0.0722 * f64::from(pixel[2])
    };
    let mut candidates = Vec::new();
    for offset in -FIXED_CROSS_SECTION_RADIUS_PX..=FIXED_CROSS_SECTION_RADIUS_PX {
        let offset = f64::from(offset);
        let sx = seam.0 + side * normal.0 * distance + tangent.0 * offset;
        let sy = seam.1 + side * normal.1 * distance + tangent.1 * offset;
        let ix = sx.round() as i32;
        let iy = sy.round() as i32;
        if ix < 1 || iy < 1 || ix + 1 >= width as i32 || iy + 1 >= height as i32 {
            continue;
        }
        let stencil = [
            (ix, iy),
            (ix - 1, iy),
            (ix + 1, iy),
            (ix, iy - 1),
            (ix, iy + 1),
        ];
        if stencil.iter().any(|&(x, y)| {
            let index = y as usize * width as usize + x as usize;
            coverage.as_raw().get(index).copied().unwrap_or(0) == 0
                || ownership.get(index).copied() != Some(expected_owner)
        }) {
            continue;
        }
        let gx = luminance(*image.get_pixel((ix + 1) as u32, iy as u32))
            - luminance(*image.get_pixel((ix - 1) as u32, iy as u32));
        let gy = luminance(*image.get_pixel(ix as u32, (iy + 1) as u32))
            - luminance(*image.get_pixel(ix as u32, (iy - 1) as u32));
        let strength = gx.hypot(gy) * 0.5;
        if strength < 0.15 {
            continue;
        }
        let tangent_alignment = (gx * tangent.0 + gy * tangent.1).abs() / gx.hypot(gy);
        if tangent_alignment < 30.0f64.to_radians().cos() {
            continue;
        }
        candidates.push(BoundaryEdge {
            offset,
            strength,
            gradient: (gx, gy),
            pixel: (ix as u32, iy as u32),
        });
    }
    candidates.sort_by(|left, right| {
        right
            .strength
            .total_cmp(&left.strength)
            .then_with(|| left.offset.total_cmp(&right.offset))
    });
    let Some(best) = candidates.first().cloned() else {
        return Err(degradation::BOUNDARY_LOW_CONTRAST);
    };
    if candidates.iter().skip(1).any(|candidate| {
        (candidate.offset - best.offset).abs() >= 3.0 && candidate.strength > best.strength * 0.8
    }) {
        return Err(degradation::BOUNDARY_STROKE_NOT_UNIQUE);
    }
    Ok(best)
}

fn boundary_pair_error(
    image: &Rgb32FImage,
    coverage: &GrayImage,
    ownership: &[u16],
    seam: (f64, f64),
    normal: (f64, f64),
    left_owner: u16,
    right_owner: u16,
) -> Result<(BoundaryEdge, BoundaryEdge), &'static str> {
    let distance = 4.0;
    let left = boundary_edge_on_side(
        image, coverage, ownership, seam, normal, -1.0, distance, left_owner,
    )?;
    let right = boundary_edge_on_side(
        image,
        coverage,
        ownership,
        seam,
        normal,
        1.0,
        distance,
        right_owner,
    )?;
    let left_length = left.gradient.0.hypot(left.gradient.1);
    let right_length = right.gradient.0.hypot(right.gradient.1);
    let cosine = (left.gradient.0 * right.gradient.0 + left.gradient.1 * right.gradient.1)
        / (left_length * right_length);
    let angle_error = cosine.clamp(-1.0, 1.0).acos().to_degrees();
    if angle_error > 10.0 {
        return Err(degradation::BOUNDARY_ORIENTATION_MISMATCH);
    }
    Ok((left, right))
}

fn source_contains_world(
    source: &QualitySource,
    world: (f64, f64),
    residual: &ResidualWarp,
) -> bool {
    quality_gate::map_world_to_source(Point2::new(world.0, world.1), &source.geometry, residual)
        .ok()
        .is_some_and(|point| {
            point.x >= 0.0
                && point.y >= 0.0
                && point.x < f64::from(source.dimensions.0)
                && point.y < f64::from(source.dimensions.1)
        })
}

fn boundary_rejection_priority(reason: &str) -> u8 {
    match reason {
        degradation::BOUNDARY_PROVENANCE_UNAVAILABLE => 4,
        degradation::BOUNDARY_ORIENTATION_MISMATCH => 3,
        degradation::BOUNDARY_STROKE_NOT_UNIQUE => 2,
        degradation::BOUNDARY_LOW_CONTRAST => 1,
        _ => 0,
    }
}
