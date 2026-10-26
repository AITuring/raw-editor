//! Virtual-tile residual fitting.

use super::*;

#[derive(Debug, Clone)]
pub(crate) struct VirtualTileResidualFit {
    pub(crate) measured_homography: Matrix3<f64>,
    pub(crate) inlier_indices: Vec<usize>,
    pub(crate) median_error_px: f64,
    pub(crate) spatial_support: f64,
    pub(crate) model: &'static str,
}

pub(crate) fn residual_distribution(
    mut values: Vec<f64>,
) -> stack_report::ResidualDistributionRecord {
    values.retain(|value| value.is_finite());
    values.sort_by(f64::total_cmp);
    if values.is_empty() {
        return Default::default();
    }
    let at = |quantile: f64| {
        let index = ((values.len() - 1) as f64 * quantile).round() as usize;
        values[index]
    };
    stack_report::ResidualDistributionRecord {
        count: values.len(),
        min: values[0],
        p10: at(0.10),
        median: at(0.50),
        p90: at(0.90),
        max: values[values.len() - 1],
    }
}

pub(crate) fn virtual_tile_measured_residuals(
    points: &[(Point2<f64>, Point2<f64>)],
    initial: &Matrix3<f64>,
) -> Vec<(usize, nalgebra::Vector2<f64>)> {
    points
        .iter()
        .enumerate()
        .filter_map(|(index, (source, target))| {
            let predicted = transformed_point(initial, *source)?;
            let residual = target - predicted;
            residual
                .iter()
                .all(|value| value.is_finite())
                .then_some((index, residual))
        })
        .collect()
}

pub(crate) fn virtual_tile_residual_candidate(
    correction: &Matrix3<f64>,
    initial: &Matrix3<f64>,
    measured_points: &[(Point2<f64>, Point2<f64>)],
    left_dimensions: (u32, u32),
    right_dimensions: (u32, u32),
    model: &'static str,
    inlier_threshold_px: f64,
) -> Option<VirtualTileResidualFit> {
    // Both matrices map left -> right, so a correction measured in right-tile
    // coordinates composes on the left: H_measured = H_residual * H_initial.
    let measured_homography = correction * initial;
    let inlier_indices =
        symmetric_inlier_indices(&measured_homography, measured_points, inlier_threshold_px);
    let inlier_points = inlier_indices
        .iter()
        .map(|&index| measured_points[index])
        .collect::<Vec<_>>();
    let median_error_px = median_symmetric_error(&measured_homography, &inlier_points);
    let spatial_support = station_relation_spatial_support(
        &inlier_points,
        &measured_homography,
        left_dimensions,
        right_dimensions,
    );
    Some(VirtualTileResidualFit {
        measured_homography,
        inlier_indices,
        median_error_px,
        spatial_support,
        model,
    })
}

pub(crate) fn virtual_tile_residual_fit_is_better(
    candidate: &VirtualTileResidualFit,
    current: &VirtualTileResidualFit,
) -> bool {
    candidate.inlier_indices.len() > current.inlier_indices.len()
        || (candidate.inlier_indices.len() == current.inlier_indices.len()
            && (candidate.median_error_px < current.median_error_px
                || (candidate.median_error_px == current.median_error_px
                    && candidate
                        .measured_homography
                        .iter()
                        .map(|value| value.to_bits())
                        .cmp(
                            current
                                .measured_homography
                                .iter()
                                .map(|value| value.to_bits()),
                        )
                        .is_lt())))
}

/// Whole left tile as the bounded-correction evidence region.
#[cfg(test)]
pub(crate) fn virtual_tile_full_tile_bounds(left_dimensions: (u32, u32)) -> (f64, f64, f64, f64) {
    (
        0.0,
        0.0,
        f64::from(left_dimensions.0),
        f64::from(left_dimensions.1),
    )
}

/// The correction may move the evidence region's corners by at most
/// `VIRTUAL_TILE_DIRECT_MAX_RESIDUAL_DISPLACEMENT_PX`, differ from its centre's
/// translation by at most `VIRTUAL_TILE_DIRECT_MAX_NON_TRANSLATION_RESIDUAL_PX`,
/// and keep its scale within the residual scale range. `evidence_bounds` is
/// `(min_x, min_y, max_x, max_y)` in left-tile pixels.
pub(crate) fn virtual_tile_residual_correction_is_bounded(
    correction: &Matrix3<f64>,
    initial: &Matrix3<f64>,
    evidence_bounds: (f64, f64, f64, f64),
) -> bool {
    let (min_x, min_y, max_x, max_y) = evidence_bounds;
    let source_samples = [
        Point2::new(min_x, min_y),
        Point2::new(max_x, min_y),
        Point2::new(max_x, max_y),
        Point2::new(min_x, max_y),
        Point2::new((min_x + max_x) * 0.5, (min_y + max_y) * 0.5),
    ];
    let Some(displacements) = source_samples
        .into_iter()
        .map(|source| {
            let predicted = transformed_point(initial, source)?;
            let corrected = transformed_point(correction, predicted)?;
            Some(corrected - predicted)
        })
        .collect::<Option<Vec<_>>>()
    else {
        return false;
    };
    let center = displacements[4];
    // The correction's scale is its local one where the evidence lies. The
    // upper-left block of a projective correction is not a Jacobian anywhere
    // but at the origin: at 1/8 on langyuan-10 it read 1.035 for a correction
    // whose local scale over its 900px evidence region stayed near 1.0.
    let local_scales = source_samples
        .into_iter()
        .map(|source| {
            let predicted = transformed_point(initial, source)?;
            let jacobian = virtual_tile_local_jacobian(correction, predicted)?;
            Some(jacobian.determinant().abs().sqrt())
        })
        .collect::<Option<Vec<_>>>();
    displacements.iter().all(|displacement| {
        displacement.norm() <= VIRTUAL_TILE_DIRECT_MAX_RESIDUAL_DISPLACEMENT_PX
            && (*displacement - center).norm()
                <= VIRTUAL_TILE_DIRECT_MAX_NON_TRANSLATION_RESIDUAL_PX
    }) && local_scales.is_some_and(|scales| {
        scales.iter().all(|scale| {
            (VIRTUAL_TILE_DIRECT_RESIDUAL_SCALE_MIN..=VIRTUAL_TILE_DIRECT_RESIDUAL_SCALE_MAX)
                .contains(scale)
        })
    })
}

pub(crate) fn virtual_tile_robust_translation_residual(
    measured_points: &[(Point2<f64>, Point2<f64>)],
    initial: &Matrix3<f64>,
    left_dimensions: (u32, u32),
    right_dimensions: (u32, u32),
    inlier_threshold_px: f64,
) -> Option<VirtualTileResidualFit> {
    let residuals = virtual_tile_measured_residuals(measured_points, initial);
    let mut hypotheses = residuals
        .iter()
        .map(|(_, residual)| *residual)
        .collect::<Vec<_>>();
    hypotheses.sort_by(|left, right| {
        left.x
            .total_cmp(&right.x)
            .then_with(|| left.y.total_cmp(&right.y))
    });
    hypotheses.dedup_by(|left, right| left.x == right.x && left.y == right.y);

    let mut best = None::<VirtualTileResidualFit>;
    for residual in hypotheses {
        let correction = Matrix3::new(1.0, 0.0, residual.x, 0.0, 1.0, residual.y, 0.0, 0.0, 1.0);
        let candidate = virtual_tile_residual_candidate(
            &correction,
            initial,
            measured_points,
            left_dimensions,
            right_dimensions,
            "translation",
            inlier_threshold_px,
        )?;
        if best
            .as_ref()
            .is_none_or(|current| virtual_tile_residual_fit_is_better(&candidate, current))
        {
            best = Some(candidate);
        }
    }

    let best = best?;
    let best_indices = best.inlier_indices.iter().copied().collect::<HashSet<_>>();
    let mut dx = residuals
        .iter()
        .filter(|(index, _)| best_indices.contains(index))
        .map(|(_, residual)| residual.x)
        .collect::<Vec<_>>();
    let mut dy = residuals
        .iter()
        .filter(|(index, _)| best_indices.contains(index))
        .map(|(_, residual)| residual.y)
        .collect::<Vec<_>>();
    dx.sort_by(f64::total_cmp);
    dy.sort_by(f64::total_cmp);
    if dx.is_empty() || dy.is_empty() {
        return Some(best);
    }
    let correction = Matrix3::new(
        1.0,
        0.0,
        dx[dx.len() / 2],
        0.0,
        1.0,
        dy[dy.len() / 2],
        0.0,
        0.0,
        1.0,
    );
    let refined = virtual_tile_residual_candidate(
        &correction,
        initial,
        measured_points,
        left_dimensions,
        right_dimensions,
        "translation",
        inlier_threshold_px,
    )?;
    if virtual_tile_residual_fit_is_better(&refined, &best) {
        Some(refined)
    } else {
        Some(best)
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn virtual_tile_refit_residual_affine(
    seed: Matrix3<f64>,
    residual_points: &[(Point2<f64>, Point2<f64>)],
    measured_points: &[(Point2<f64>, Point2<f64>)],
    initial: &Matrix3<f64>,
    left_dimensions: (u32, u32),
    right_dimensions: (u32, u32),
    inlier_threshold_px: f64,
    evidence_bounds: (f64, f64, f64, f64),
) -> Option<VirtualTileResidualFit> {
    let mut correction = seed;
    let mut fit = virtual_tile_residual_candidate(
        &correction,
        initial,
        measured_points,
        left_dimensions,
        right_dimensions,
        "affine",
        inlier_threshold_px,
    )?;
    for _ in 0..2 {
        let points = fit
            .inlier_indices
            .iter()
            .map(|&index| residual_points[index])
            .collect::<Vec<_>>();
        let Some(refitted) = estimate_affine(&points) else {
            break;
        };
        if !virtual_tile_residual_correction_is_bounded(&refitted, initial, evidence_bounds) {
            break;
        }
        correction = refitted;
        fit = virtual_tile_residual_candidate(
            &correction,
            initial,
            measured_points,
            left_dimensions,
            right_dimensions,
            "affine",
            inlier_threshold_px,
        )?;
    }
    virtual_tile_residual_correction_is_bounded(&correction, initial, evidence_bounds)
        .then_some(fit)
}

/// How the residual model chooses among its passing stages.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum VirtualTileResidualSelection {
    /// Stop at the lowest-DOF stage that passes (single-level rule).
    LowestComplexity,
    /// Evaluate every stage and keep the passing one with the most measured
    /// inliers; on an inlier tie a strictly lower median error wins, otherwise
    /// the lower-DOF stage stays. A coarse-to-fine level must carry
    /// the seed's rotation, scale and perspective error forward: a translation
    /// that explains only the overlap's central band would leave the finer level
    /// a correction it cannot bound.
    MostInliers,
}

#[derive(Debug)]
pub(crate) struct VirtualTileResidualModelOutcome {
    pub(crate) fit: Option<VirtualTileResidualFit>,
    pub(crate) max_translation_consensus: usize,
    pub(crate) translation: stack_report::ResidualModelStageRecord,
    pub(crate) affine: stack_report::ResidualModelStageRecord,
    pub(crate) projective: stack_report::ResidualModelStageRecord,
}

pub(crate) fn residual_model_stage(
    fit: Option<&VirtualTileResidualFit>,
    bounded_correction: Option<bool>,
) -> stack_report::ResidualModelStageRecord {
    let candidate_inliers = fit.map_or(0, |fit| fit.inlier_indices.len());
    let final_support = fit.map_or(0.0, |fit| fit.spatial_support);
    let rejection_reason = if bounded_correction == Some(false) {
        "correction_out_of_bounds"
    } else if fit.is_none() {
        "no_fit"
    } else if candidate_inliers < STATION_RELATION_MIN_INLIERS {
        "insufficient_inliers"
    } else if final_support < STATION_RELATION_MIN_SPATIAL_SUPPORT {
        "insufficient_support"
    } else {
        "accepted"
    };
    stack_report::ResidualModelStageRecord {
        candidate_inliers,
        final_support,
        bounded_correction,
        rejection_reason: rejection_reason.to_string(),
    }
}

pub(crate) fn residual_model_not_run_stage() -> stack_report::ResidualModelStageRecord {
    stack_report::ResidualModelStageRecord {
        rejection_reason: "not_run".to_string(),
        ..Default::default()
    }
}

/// Residual model fit whose inlier tolerance is the requirement-6.3 world
/// pixel bound itself; the points are native tile pixels and the whole left
/// tile bounds the correction (the single-level rule).
#[cfg(test)]
pub(crate) fn virtual_tile_fit_measured_residual_model(
    measured_points: &[(Point2<f64>, Point2<f64>)],
    initial: &Matrix3<f64>,
    left_dimensions: (u32, u32),
    right_dimensions: (u32, u32),
) -> VirtualTileResidualModelOutcome {
    virtual_tile_fit_measured_residual_model_with_threshold(
        measured_points,
        initial,
        left_dimensions,
        right_dimensions,
        STATION_RELATION_MAX_MEDIAN_ERROR_PX,
        virtual_tile_full_tile_bounds(left_dimensions),
        VirtualTileResidualSelection::LowestComplexity,
        StationOverlapMeasure::Rectangles,
    )
}

/// Re-measure a stage fit's requirement-6.4 support over `overlap`. The fits
/// themselves rank hypotheses by inliers and error only, so the (costlier)
/// covered overlap is measured once per stage rather than per hypothesis.
pub(crate) fn virtual_tile_fit_with_overlap_support(
    mut fit: VirtualTileResidualFit,
    measured_points: &[(Point2<f64>, Point2<f64>)],
    left_dimensions: (u32, u32),
    right_dimensions: (u32, u32),
    overlap: StationOverlapMeasure<'_>,
) -> VirtualTileResidualFit {
    if matches!(overlap, StationOverlapMeasure::Covered { .. }) {
        let inlier_points = fit
            .inlier_indices
            .iter()
            .map(|&index| measured_points[index])
            .collect::<Vec<_>>();
        fit.spatial_support = station_relation_spatial_support_with(
            &inlier_points,
            &fit.measured_homography,
            left_dimensions,
            right_dimensions,
            overlap,
        );
    }
    fit
}

/// Residual model fit in the pixels of one probe level. `inlier_threshold_px`
/// is a symmetric error in those pixels, so the finest pyramid level passes the
/// requirement-6.3 bound divided by its area-averaging factor. The correction
/// bounds are evaluated over `evidence_bounds` (left-tile pixels), the region
/// the probes actually measured; each stage's support over `overlap`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn virtual_tile_fit_measured_residual_model_with_threshold(
    measured_points: &[(Point2<f64>, Point2<f64>)],
    initial: &Matrix3<f64>,
    left_dimensions: (u32, u32),
    right_dimensions: (u32, u32),
    inlier_threshold_px: f64,
    evidence_bounds: (f64, f64, f64, f64),
    selection: VirtualTileResidualSelection,
    overlap: StationOverlapMeasure<'_>,
) -> VirtualTileResidualModelOutcome {
    let with_support = |fit: VirtualTileResidualFit| {
        virtual_tile_fit_with_overlap_support(
            fit,
            measured_points,
            left_dimensions,
            right_dimensions,
            overlap,
        )
    };
    let translation = virtual_tile_robust_translation_residual(
        measured_points,
        initial,
        left_dimensions,
        right_dimensions,
        inlier_threshold_px,
    )
    .map(with_support);
    let max_translation_consensus = translation
        .as_ref()
        .map_or(0, |fit| fit.inlier_indices.len());
    let translation_correction = translation.as_ref().and_then(|fit| {
        initial
            .try_inverse()
            .map(|initial_inverse| fit.measured_homography * initial_inverse)
    });
    let translation_bounded = translation_correction.as_ref().map(|correction| {
        virtual_tile_residual_correction_is_bounded(correction, initial, evidence_bounds)
    });
    let translation_stage = residual_model_stage(translation.as_ref(), translation_bounded);
    if selection == VirtualTileResidualSelection::LowestComplexity
        && translation_stage.rejection_reason == "accepted"
    {
        return VirtualTileResidualModelOutcome {
            fit: translation,
            max_translation_consensus,
            translation: translation_stage,
            affine: residual_model_not_run_stage(),
            projective: residual_model_not_run_stage(),
        };
    }

    // Escalate model complexity only when the lower-DOF correction cannot pass
    // the unchanged inlier/support/bounds gates (or, for a coarse-to-fine
    // level, evaluate every stage). The fit uses predicted target points as
    // coordinates, but every accepted inlier is recomputed from the original
    // measured source/target pair under H_residual * H_initial.
    let mut indexed_residual_points = measured_points
        .iter()
        .enumerate()
        .filter_map(|(index, (source, target))| {
            transformed_point(initial, *source).map(|predicted| (index, predicted, *target))
        })
        .collect::<Vec<_>>();
    indexed_residual_points.sort_by(|left, right| {
        left.1
            .y
            .total_cmp(&right.1.y)
            .then_with(|| left.1.x.total_cmp(&right.1.x))
            .then_with(|| left.2.y.total_cmp(&right.2.y))
            .then_with(|| left.2.x.total_cmp(&right.2.x))
    });
    let sorted_pairs = indexed_residual_points
        .iter()
        .map(|(_, predicted, target)| (*predicted, *target))
        .collect::<Vec<_>>();
    let residual_points = measured_points
        .iter()
        .map(|(source, target)| {
            (
                transformed_point(initial, *source).unwrap_or(*source),
                *target,
            )
        })
        .collect::<Vec<_>>();

    // The single-level rule seeds the affine with the source-stage tolerance;
    // a pyramid level seeds it at its own precision so its spurious peaks
    // cannot pull the seed out of bounds.
    let affine_seed_threshold = match selection {
        VirtualTileResidualSelection::LowestComplexity => FOCUS_MODEL_INLIER_THRESHOLD,
        VirtualTileResidualSelection::MostInliers => inlier_threshold_px,
    };
    let affine_seed = robust_transform_fit_with_threshold(
        &sorted_pairs,
        3,
        0xD1EC_7A11_AFF1_4E01,
        affine_seed_threshold,
        estimate_affine,
    );
    let affine_seed_bounded = affine_seed.as_ref().map(|seed| {
        virtual_tile_residual_correction_is_bounded(&seed.transform, initial, evidence_bounds)
    });
    let affine = affine_seed
        .filter(|_| affine_seed_bounded == Some(true))
        .and_then(|seed| {
            virtual_tile_refit_residual_affine(
                seed.transform,
                &residual_points,
                measured_points,
                initial,
                left_dimensions,
                right_dimensions,
                inlier_threshold_px,
                evidence_bounds,
            )
        })
        .map(with_support);
    let affine_bounded = affine
        .as_ref()
        .map(|fit| {
            initial.try_inverse().is_some_and(|initial_inverse| {
                let correction = fit.measured_homography * initial_inverse;
                virtual_tile_residual_correction_is_bounded(&correction, initial, evidence_bounds)
            })
        })
        .or(affine_seed_bounded);
    let affine_stage = residual_model_stage(affine.as_ref(), affine_bounded);
    if selection == VirtualTileResidualSelection::LowestComplexity
        && affine_stage.rejection_reason == "accepted"
    {
        return VirtualTileResidualModelOutcome {
            fit: affine,
            max_translation_consensus,
            translation: translation_stage,
            affine: affine_stage,
            projective: residual_model_not_run_stage(),
        };
    }

    let projective_seed = processing::find_homography_ransac_points_stable_with_min_inliers(
        &sorted_pairs,
        inlier_threshold_px,
        STATION_RELATION_MIN_INLIERS,
    );
    let projective_correction = projective_seed.as_ref().map(|(seed, seed_inliers)| {
        let seed_points = seed_inliers
            .iter()
            .map(|&index| sorted_pairs[index])
            .collect::<Vec<_>>();
        processing::compute_homography(&seed_points).unwrap_or(*seed)
    });
    let projective_bounded = projective_correction.as_ref().map(|correction| {
        virtual_tile_residual_correction_is_bounded(correction, initial, evidence_bounds)
    });
    let projective = projective_correction
        .filter(|_| projective_bounded == Some(true))
        .and_then(|correction| {
            virtual_tile_residual_candidate(
                &correction,
                initial,
                measured_points,
                left_dimensions,
                right_dimensions,
                "small_projective",
                inlier_threshold_px,
            )
        })
        .map(with_support);
    let projective_stage = residual_model_stage(projective.as_ref(), projective_bounded);
    let fit = match selection {
        VirtualTileResidualSelection::LowestComplexity => (projective_stage.rejection_reason
            == "accepted")
            .then_some(projective)
            .flatten(),
        VirtualTileResidualSelection::MostInliers => {
            // Lowest DOF first: a later stage replaces it with strictly more
            // measured inliers, or with as many and a strictly lower median
            // error. Under a coarse tolerance a translation and an affine can
            // explain the same points; only the lower error shows which one
            // carries the seed's rotation and scale remainder.
            let mut chosen = None::<VirtualTileResidualFit>;
            for (fit, stage) in [
                (translation, &translation_stage),
                (affine, &affine_stage),
                (projective, &projective_stage),
            ] {
                let Some(fit) = fit.filter(|_| stage.rejection_reason == "accepted") else {
                    continue;
                };
                if chosen.as_ref().is_none_or(|current| {
                    fit.inlier_indices.len() > current.inlier_indices.len()
                        || (fit.inlier_indices.len() == current.inlier_indices.len()
                            && fit.median_error_px < current.median_error_px)
                }) {
                    chosen = Some(fit);
                }
            }
            chosen
        }
    };
    VirtualTileResidualModelOutcome {
        fit,
        max_translation_consensus,
        translation: translation_stage,
        affine: affine_stage,
        projective: projective_stage,
    }
}
