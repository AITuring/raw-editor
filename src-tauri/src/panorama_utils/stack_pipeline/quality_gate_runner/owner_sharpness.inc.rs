/// Build the owner-sharpness criterion from the exact planes retained by the
/// compositor.  Keeping this construction here makes the runtime report and
/// Property 71 exercise the same statistic, threshold, and evidence policy.
pub(crate) struct OwnerSharpnessCriterionInput<'a> {
    pub(crate) coverage: &'a GrayImage,
    pub(crate) confidence: &'a [f32],
    pub(crate) textured: Option<&'a [u8]>,
    pub(crate) shortfall: Option<&'a [u8]>,
    pub(crate) disagreement: Option<&'a [u8]>,
    pub(crate) unresolved_pixel_count: u64,
    pub(crate) world_origin: (f64, f64),
    pub(crate) unmeasurable: &'a mut Vec<UnmeasurableRecord>,
}

pub(crate) fn build_owner_sharpness_criterion(
    input: OwnerSharpnessCriterionInput<'_>,
) -> Criterion {
    let mut criterion = Criterion::new(
        "owner_sharpness_coverage",
        quality_gate::OWNER_SHARPNESS_COVERAGE_MIN,
    );
    let confidence_stats = input
        .textured
        .and_then(|textured| confidence_coverage_stats(input.coverage, input.confidence, textured));
    let owner_stats = match (input.textured, input.shortfall, input.disagreement) {
        (Some(textured), Some(shortfall), Some(disagreement)) => owner_sharpness_stats(
            input.coverage.as_raw(),
            textured,
            shortfall,
            disagreement,
            input.unresolved_pixel_count,
        )
        .ok(),
        _ => None,
    };
    match (confidence_stats, owner_stats) {
        (Some(confidence_stats), Some(stats)) => {
            criterion.textured_pixel_count = Some(stats.textured_pixels);
            criterion.excluded_flat_pixel_count = Some(stats.flat_pixels);
            criterion.textured_shortfall_ratio = (stats.textured_pixels > 0)
                .then_some(stats.textured_shortfall as f64 / stats.textured_pixels as f64);
            criterion.flat_shortfall_ratio = (stats.flat_pixels > 0)
                .then_some(stats.flat_shortfall as f64 / stats.flat_pixels as f64);
            criterion.shortfall_histogram = stats.shortfall_histogram.clone();
            criterion.disagreement_veto_count = stats.disagreement_veto_count;
            criterion.unresolved_pixel_count = stats.unresolved_pixel_count;
            criterion.textured_low_confidence_ratio = (confidence_stats.textured_pixels > 0
                && confidence_stats.textured_unknown == 0)
                .then_some(
                    confidence_stats.textured_low as f64 / confidence_stats.textured_pixels as f64,
                );
            criterion.flat_low_confidence_ratio = (confidence_stats.flat_pixels > 0
                && confidence_stats.flat_unknown == 0)
                .then_some(confidence_stats.flat_low as f64 / confidence_stats.flat_pixels as f64);
            if stats.unresolved_pixel_count > 0 {
                criterion.miss_count(
                    degradation::OWNER_REVERSE_LOOKUP_UNRESOLVED,
                    stats.unresolved_pixel_count,
                );
                input.unmeasurable.push(UnmeasurableRecord {
                    criterion: criterion.name.to_string(),
                    world: WorldPoint {
                        x: input.world_origin.0,
                        y: input.world_origin.1,
                    },
                    reason: degradation::OWNER_REVERSE_LOOKUP_UNRESOLVED.to_string(),
                    category: UnmeasurableCategory::Technical,
                });
            }
            if stats.textured_pixels == 0 {
                criterion.miss(degradation::TEXTURED_PIXEL_PLANE_UNAVAILABLE);
                input.unmeasurable.push(UnmeasurableRecord {
                    criterion: criterion.name.to_string(),
                    world: WorldPoint {
                        x: input.world_origin.0,
                        y: input.world_origin.1,
                    },
                    reason: degradation::TEXTURED_PIXEL_PLANE_UNAVAILABLE.to_string(),
                    category: classify_unmeasurable_reason(
                        degradation::TEXTURED_PIXEL_PLANE_UNAVAILABLE,
                    ),
                });
            } else {
                let value = 1.0 - stats.textured_shortfall as f64 / stats.textured_pixels as f64;
                criterion.push(input.world_origin, value, "", value >= criterion.threshold);
            }
        }
        _ => {
            criterion.miss(degradation::TEXTURED_PIXEL_PLANE_UNAVAILABLE);
            input.unmeasurable.push(UnmeasurableRecord {
                criterion: criterion.name.to_string(),
                world: WorldPoint {
                    x: input.world_origin.0,
                    y: input.world_origin.1,
                },
                reason: degradation::TEXTURED_PIXEL_PLANE_UNAVAILABLE.to_string(),
                category: classify_unmeasurable_reason(
                    degradation::TEXTURED_PIXEL_PLANE_UNAVAILABLE,
                ),
            });
        }
    }
    criterion
}
