//! Virtual-tile native station polishing.

use super::*;

/// Coarse pixels establish a search model. Final camera geometry is measured
/// on native pixels: a one-level-pixel reverse check at factor four otherwise
/// admits four-native-pixel errors and biases long projective tree chains.
pub(crate) fn virtual_tile_native_station_polish(
    left: &(GrayImage, stitching::CoverageMask),
    right: &(GrayImage, stitching::CoverageMask),
    seed: &MatchInfo,
    diagnostic: &mut stack_report::StationRelationCandidateRecord,
) -> Option<MatchInfo> {
    let inverse = seed.homography.try_inverse()?;
    let step = (seed.points.len() / 1000).max(1);
    let supported = |mask: &stitching::CoverageMask, point: Point2<f64>| {
        let (w, h) = mask.dimensions();
        let radius = 40i32;
        let (x, y) = (point.x.round() as i32, point.y.round() as i32);
        x >= radius
            && y >= radius
            && x + radius < w as i32
            && y + radius < h as i32
            && (-radius..=radius).step_by(8).all(|dy| {
                (-radius..=radius)
                    .step_by(8)
                    .all(|dx| mask.is_covered((x + dx) as u32, (y + dy) as u32))
            })
    };
    let points = seed
        .points
        .par_iter()
        .enumerate()
        .filter(|(index, _)| index % step == 0)
        .filter_map(|(_, &(source, target))| {
            if !supported(&left.1, source) || !supported(&right.1, target) {
                return None;
            }
            let measured = residual_warp::verify_round_trip_match_with_template(
                &left.0,
                &right.0,
                &seed.homography,
                &inverse,
                source,
                31,
            )?;
            supported(&right.1, measured.forward).then_some((source, measured.forward))
        })
        .collect::<Vec<_>>();
    diagnostic.failure_stage = "insufficient_native_round_trip_matches".to_string();
    diagnostic.bidirectional_match_count = points.len();
    diagnostic.measurement_factor = 1;
    if points.len() < STATION_RELATION_MIN_INLIERS {
        return None;
    }
    let (homography, indices) = virtual_tile_polish_fit(
        &points,
        VIRTUAL_TILE_POLISH_INLIER_THRESHOLD_NATIVE_PX,
        left.0.dimensions(),
    )?;
    let points = indices
        .into_iter()
        .map(|index| points[index])
        .collect::<Vec<_>>();
    diagnostic.fitted_inliers = points.len();
    diagnostic.hull_support = station_relation_spatial_support_with(
        &points,
        &homography,
        left.0.dimensions(),
        right.0.dimensions(),
        StationOverlapMeasure::Covered {
            source: &left.1,
            target: &right.1,
            factor: 1,
        },
    );
    diagnostic.residual_magnitude_px = residual_distribution(match homography.try_inverse() {
        Some(inverse) => points
            .iter()
            .map(|(a, b)| symmetric_point_error(&homography, &inverse, *a, *b))
            .collect(),
        None => Vec::new(),
    });
    diagnostic.failure_stage = "insufficient_native_polished_support".to_string();
    if diagnostic.hull_support < STATION_RELATION_MIN_SPATIAL_SUPPORT
        || median_symmetric_error(&homography, &points) > STATION_RELATION_MAX_MEDIAN_ERROR_PX
    {
        return None;
    }
    diagnostic.residual_model = "native_polished_projective".to_string();
    diagnostic.failure_stage = "fitted".to_string();
    Some(MatchInfo {
        homography,
        inliers: points.len(),
        points,
        canonical_homography: None,
        ..seed.clone()
    })
}

// ---------------------------------------------------------------------------
// Finest-octave polish
//
// The bounded 13-sample probe is a good *seed* but a poor final instrument on
// the finest octave: on langyuan-10 only 23–30% of its 1/4 bidirectional
// matches agreed with an independent relation, and the ones that did covered
// 16–19% of the 0→2 overlap. Matching 15×15 templates of the left plane against
// the right plane resampled through the current relation (so the template and
// the window share one geometry) gave 846–1011 inliers at a 0.9–1.1px one-way
// median, and 0→2 reached 0.22–0.28 support at a 1.2–1.9px one-way median once
// the inlier tolerance admitted the overlap's non-homographic remainder.
// Requirement 6.3 bounds the inliers' *median*; the solver re-checks it.
// ---------------------------------------------------------------------------

pub(crate) const VIRTUAL_TILE_POLISH_TEMPLATE_HALF: i32 = 7;
pub(crate) const VIRTUAL_TILE_POLISH_GRID_STEP_PX: f64 = 16.0;
/// Search radii of the two polish passes, in finest-plane pixels. The first
/// absorbs what the coarser octaves left; the second re-matches around it.
pub(crate) const VIRTUAL_TILE_POLISH_SEARCH_PASSES: [i32; 2] = [8, 4];
pub(crate) const VIRTUAL_TILE_POLISH_MIN_NCC: f64 = 0.6;
pub(crate) const VIRTUAL_TILE_POLISH_MIN_PEAK_MARGIN: f64 = 0.03;
/// Smallest template standard deviation in 8-bit levels.
pub(crate) const VIRTUAL_TILE_POLISH_MIN_TEMPLATE_STD: f64 = 3.0;
/// Symmetric inlier tolerance of the polished relation in native pixels: both
/// one-way errors within the requirement-6.3 3.0 world pixels. (Twice that let
/// the refits drift to a looser model: 0→2 read a 3.43px symmetric median,
/// against 2.09px at 0.29 support with this tolerance.)
pub(crate) const VIRTUAL_TILE_POLISH_INLIER_THRESHOLD_NATIVE_PX: f64 =
    std::f64::consts::SQRT_2 * STATION_RELATION_MAX_MEDIAN_ERROR_PX;

#[derive(Default)]
pub(crate) struct VirtualTilePolishCounts {
    pub(crate) probes: usize,
    pub(crate) textured: usize,
    pub(crate) matched: usize,
}

/// Zero-mean normalised cross-correlation of two equally long sample runs.
pub(crate) fn virtual_tile_ncc(left: &[f64], right: &[f64]) -> Option<f64> {
    let count = left.len() as f64;
    if left.len() != right.len() || left.is_empty() {
        return None;
    }
    let left_mean = left.iter().sum::<f64>() / count;
    let right_mean = right.iter().sum::<f64>() / count;
    let mut product = 0.0;
    let mut left_energy = 0.0;
    let mut right_energy = 0.0;
    for (&l, &r) in left.iter().zip(right) {
        let (l, r) = (l - left_mean, r - right_mean);
        product += l * r;
        left_energy += l * l;
        right_energy += r * r;
    }
    let denominator = (left_energy * right_energy).sqrt();
    (denominator > f64::EPSILON).then(|| product / denominator)
}

/// Best integer offset of `template` inside a square `window` of side
/// `2·(half + search) + 1`, its NCC, and the best NCC at least two pixels away.
#[allow(clippy::type_complexity)]
pub(crate) fn virtual_tile_best_window_offset(
    template: &[f64],
    window: &[f64],
    half: i32,
    search: i32,
) -> Option<((i32, i32), f64, f64, Vec<f64>)> {
    let side = (2 * (half + search) + 1) as usize;
    let template_side = (2 * half + 1) as usize;
    let offsets = (2 * search + 1) as usize;
    let mut scores = vec![-1.0; offsets * offsets];
    let mut sample = vec![0.0; template.len()];
    for oy in 0..offsets {
        for ox in 0..offsets {
            for ty in 0..template_side {
                let row = (oy + ty) * side + ox;
                sample[ty * template_side..(ty + 1) * template_side]
                    .copy_from_slice(&window[row..row + template_side]);
            }
            if let Some(score) = virtual_tile_ncc(template, &sample) {
                scores[oy * offsets + ox] = score;
            }
        }
    }
    let (best_index, &best) =
        scores
            .iter()
            .enumerate()
            .max_by(|(left_index, left), (right_index, right)| {
                left.total_cmp(right)
                    .then_with(|| right_index.cmp(left_index))
            })?;
    let (bx, by) = ((best_index % offsets) as i32, (best_index / offsets) as i32);
    let second = scores
        .iter()
        .enumerate()
        .filter(|(index, _)| {
            let (x, y) = ((index % offsets) as i32, (index / offsets) as i32);
            (x - bx).abs() >= 2 || (y - by).abs() >= 2
        })
        .map(|(_, &score)| score)
        .fold(-1.0f64, f64::max);
    Some(((bx - search, by - search), best, second, scores))
}

/// Model-warped template correspondences `(left plane, right plane)`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn virtual_tile_polish_correspondences(
    left: &GrayImage,
    left_coverage: &stitching::CoverageMask,
    right: &GrayImage,
    right_coverage: &stitching::CoverageMask,
    left_to_right: &Matrix3<f64>,
    bounds: (f64, f64, f64, f64),
    search: i32,
    counts: &mut VirtualTilePolishCounts,
) -> Vec<(Point2<f64>, Point2<f64>)> {
    let half = VIRTUAL_TILE_POLISH_TEMPLATE_HALF;
    #[cfg(test)]
    let half = std::env::var("RAW_EDITOR_TILE_REPLAY_TEMPLATE_HALF")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(half);
    let reach = half + search;
    let (min_x, min_y, max_x, max_y) = bounds;
    let mut centers = Vec::new();
    let mut y = (min_y + f64::from(reach)).ceil();
    while y <= max_y - f64::from(reach) {
        let mut x = (min_x + f64::from(reach)).ceil();
        while x <= max_x - f64::from(reach) {
            centers.push((x as i32, y as i32));
            x += VIRTUAL_TILE_POLISH_GRID_STEP_PX;
        }
        y += VIRTUAL_TILE_POLISH_GRID_STEP_PX;
    }
    counts.probes += centers.len();
    let right_sample = |point: Point2<f64>| -> Option<f64> {
        let (x0, y0) = (point.x.floor(), point.y.floor());
        if x0 < 0.0 || y0 < 0.0 {
            return None;
        }
        let (x0, y0) = (x0 as u32, y0 as u32);
        [(x0, y0), (x0 + 1, y0), (x0, y0 + 1), (x0 + 1, y0 + 1)]
            .iter()
            .all(|&(x, y)| right_coverage.is_covered(x, y))
            .then(|| registration::sample_gray(right, point.x, point.y))
            .flatten()
    };
    // One task per centre; results are collected in centre order.
    let results = centers
        .par_iter()
        .map(|&(cx, cy)| {
            let mut template = Vec::with_capacity(((2 * half + 1) * (2 * half + 1)) as usize);
            for dy in -half..=half {
                for dx in -half..=half {
                    let (x, y) = (cx + dx, cy + dy);
                    if x < 0 || y < 0 || !left_coverage.is_covered(x as u32, y as u32) {
                        return (false, None);
                    }
                    template.push(f64::from(left.get_pixel(x as u32, y as u32)[0]));
                }
            }
            let mean = template.iter().sum::<f64>() / template.len() as f64;
            let std = (template.iter().map(|v| (v - mean).powi(2)).sum::<f64>()
                / template.len() as f64)
                .sqrt();
            if std < VIRTUAL_TILE_POLISH_MIN_TEMPLATE_STD {
                return (false, None);
            }
            let side = (2 * reach + 1) as usize;
            let mut window = Vec::with_capacity(side * side);
            for dy in -reach..=reach {
                for dx in -reach..=reach {
                    let point = Point2::new(f64::from(cx + dx), f64::from(cy + dy));
                    let Some(value) =
                        transformed_point(left_to_right, point).and_then(right_sample)
                    else {
                        return (true, None);
                    };
                    window.push(value);
                }
            }
            let Some(((ox, oy), best, second, scores)) =
                virtual_tile_best_window_offset(&template, &window, half, search)
            else {
                return (true, None);
            };
            if best < VIRTUAL_TILE_POLISH_MIN_NCC
                || best - second < VIRTUAL_TILE_POLISH_MIN_PEAK_MARGIN
                || ox.abs() >= search
                || oy.abs() >= search
            {
                return (true, None);
            }
            // Reverse: the matched warped-right content must find its way back
            // to the template centre in the left plane.
            let reverse_template = {
                let mut values = Vec::with_capacity(template.len());
                for dy in -half..=half {
                    for dx in -half..=half {
                        let row = (oy + dy + reach) as usize;
                        let column = (ox + dx + reach) as usize;
                        values.push(window[row * side + column]);
                    }
                }
                values
            };
            let back_search = 2;
            let back_reach = half + back_search;
            let mut back_window = Vec::with_capacity(((2 * back_reach + 1).pow(2)) as usize);
            for dy in -back_reach..=back_reach {
                for dx in -back_reach..=back_reach {
                    // The reverse template should land back at the original
                    // source centre.  Searching around the forward match
                    // (`cx + ox`) made the reverse offset equal to `-ox`,
                    // although `back_search` is intentionally only two
                    // pixels wide.  A valid forward correction of five
                    // pixels was therefore rejected before fitting.  Keep
                    // the reverse search in the source coordinate frame and
                    // require its offset to return to that centre.
                    let (x, y) = (cx + dx, cy + dy);
                    if x < 0 || y < 0 || !left_coverage.is_covered(x as u32, y as u32) {
                        return (true, None);
                    }
                    back_window.push(f64::from(left.get_pixel(x as u32, y as u32)[0]));
                }
            }
            let Some(((bx, by), _, _, _)) =
                virtual_tile_best_window_offset(&reverse_template, &back_window, half, back_search)
            else {
                return (true, None);
            };
            if f64::from(bx).hypot(f64::from(by)) > 1.0 {
                return (true, None);
            }
            // Separable parabolic sub-pixel peak on the forward score surface.
            let offsets = (2 * search + 1) as usize;
            let at =
                |x: i32, y: i32| scores[(y + search) as usize * offsets + (x + search) as usize];
            let parabola = |minus: f64, center: f64, plus: f64| {
                let denominator = minus - 2.0 * center + plus;
                if denominator.abs() > f64::EPSILON {
                    (0.5 * (minus - plus) / denominator).clamp(-0.5, 0.5)
                } else {
                    0.0
                }
            };
            let sx = parabola(at(ox - 1, oy), best, at(ox + 1, oy));
            let sy = parabola(at(ox, oy - 1), best, at(ox, oy + 1));
            let warped = Point2::new(f64::from(cx + ox) + sx, f64::from(cy + oy) + sy);
            let Some(target) = transformed_point(left_to_right, warped) else {
                return (true, None);
            };
            (
                true,
                Some((Point2::new(f64::from(cx), f64::from(cy)), target)),
            )
        })
        .collect::<Vec<_>>();
    counts.textured += results.iter().filter(|(textured, _)| *textured).count();
    let matches = results
        .into_iter()
        .filter_map(|(_, correspondence)| correspondence)
        .collect::<Vec<_>>();
    counts.matched += matches.len();
    matches
}

/// Robust homography over polished correspondences: RANSAC with a per-direction
/// tolerance of `threshold / √2`, then least-squares refits on the symmetric
/// inlier set.
pub(crate) fn virtual_tile_polish_fit(
    correspondences: &[(Point2<f64>, Point2<f64>)],
    threshold: f64,
    left_dimensions: (u32, u32),
) -> Option<(Matrix3<f64>, Vec<usize>)> {
    let (seed, mut inliers) = processing::find_homography_ransac_points_stable_with_min_inliers(
        correspondences,
        threshold / std::f64::consts::SQRT_2,
        STATION_RELATION_MIN_INLIERS,
    )?;
    let mut homography = seed;
    for _ in 0..3 {
        let points = inliers
            .iter()
            .map(|&index| correspondences[index])
            .collect::<Vec<_>>();
        let Some(refitted) = processing::compute_homography(&points) else {
            break;
        };
        let refitted_inliers = symmetric_inlier_indices(&refitted, correspondences, threshold);
        if refitted_inliers.len() < inliers.len() {
            break;
        }
        let converged = refitted_inliers == inliers;
        homography = refitted;
        inliers = refitted_inliers;
        if converged {
            break;
        }
    }
    let homography = normalized_homography(&homography)?;
    (inliers.len() >= STATION_RELATION_MIN_INLIERS
        && homography_preserves_focus_orientation(&homography, left_dimensions))
    .then_some((homography, inliers))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum VirtualTilePolishModel {
    Similarity,
    Affine,
    Projective,
}

impl VirtualTilePolishModel {
    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::Similarity => "polished_similarity",
            Self::Affine => "polished_affine",
            Self::Projective => "polished_projective",
        }
    }

    pub(crate) fn fit(
        self,
        correspondences: &[(Point2<f64>, Point2<f64>)],
        threshold: f64,
        left_dimensions: (u32, u32),
    ) -> Option<(Matrix3<f64>, Vec<usize>)> {
        if self == Self::Projective {
            // Preserve the retained project's exact projective RANSAC/refit
            // path. Multi-hypothesis polish adds lower-DOF competitors; it
            // must not perturb the projective evidence that already passed.
            return virtual_tile_polish_fit(correspondences, threshold, left_dimensions);
        }
        let fit = match self {
            Self::Similarity => robust_transform_fit_with_threshold(
                correspondences,
                2,
                0x504f_4c49_5348_5349,
                threshold,
                estimate_similarity,
            ),
            Self::Affine => robust_transform_fit_with_threshold(
                correspondences,
                3,
                0x504f_4c49_5348_4146,
                threshold,
                estimate_affine,
            ),
            Self::Projective => unreachable!("handled above"),
        }?;
        let homography = normalized_homography(&fit.transform)?;
        (fit.inlier_indices.len() >= STATION_RELATION_MIN_INLIERS
            && homography.try_inverse().is_some()
            && homography_preserves_focus_orientation(&homography, left_dimensions))
        .then_some((homography, fit.inlier_indices))
    }
}

#[derive(Clone)]
pub(crate) struct VirtualTilePolishHypothesis {
    pub(crate) model: VirtualTilePolishModel,
    pub(crate) homography: Matrix3<f64>,
    pub(crate) inliers: Vec<usize>,
    pub(crate) correspondences: Vec<(Point2<f64>, Point2<f64>)>,
}

pub(crate) fn virtual_tile_hypotheses_are_equivalent(
    left: &Matrix3<f64>,
    right: &Matrix3<f64>,
    dimensions: (u32, u32),
) -> bool {
    focus_transform_disagreement_px(left, right, dimensions)
        .is_some_and(|difference| difference <= 1e-6)
}

/// Polish a pyramid relation on the finest planes. Every independently
/// measurable similarity, affine, and projective model survives to the shared
/// Station_Relation gate; this stage does not let one model's scale or hull
/// result suppress another model before dense covered-overlap measurement.
#[allow(clippy::too_many_arguments)]
pub(crate) fn virtual_tile_polished_station_match(
    left_plane: &ImageInfo,
    left_coverage: &stitching::CoverageMask,
    right_plane: &ImageInfo,
    right_coverage: &stitching::CoverageMask,
    factor: u32,
    seed_native: &Matrix3<f64>,
    native_dimensions: ((u32, u32), (u32, u32)),
    diagnostic: &mut stack_report::StationRelationCandidateRecord,
) -> Vec<MatchInfo> {
    let to_native = VirtualTileMatchPlane::level_to_native(factor);
    let from_native = VirtualTileMatchPlane::native_to_level(factor);
    let threshold = VIRTUAL_TILE_POLISH_INLIER_THRESHOLD_NATIVE_PX / f64::from(factor.max(1));
    let Some(seed) = normalized_homography(&(from_native * seed_native * to_native)) else {
        diagnostic.failure_stage = "non_invertible_seed".to_string();
        return Vec::new();
    };
    let mut hypotheses = Vec::<VirtualTilePolishHypothesis>::new();
    let mut representative_counts = VirtualTilePolishCounts::default();

    for (pass_index, search) in VIRTUAL_TILE_POLISH_SEARCH_PASSES.into_iter().enumerate() {
        let seeds = if pass_index == 0 {
            vec![(None, seed)]
        } else {
            hypotheses
                .iter()
                .map(|hypothesis| (Some(hypothesis.model), hypothesis.homography))
                .collect::<Vec<_>>()
        };
        if seeds.is_empty() {
            break;
        }
        let mut next = Vec::new();
        for (restricted_model, relation) in seeds {
            diagnostic.failure_stage = "no_predicted_overlap".to_string();
            let Some(bounds) = virtual_tile_predicted_overlap_bounds(
                &relation,
                left_plane.dimensions(),
                right_plane.dimensions(),
            ) else {
                continue;
            };
            diagnostic.predicted_overlap_area_px = diagnostic.predicted_overlap_area_px.max(
                station_relation_overlap_area(
                    &relation,
                    left_plane.dimensions(),
                    right_plane.dimensions(),
                    true,
                )
                .unwrap_or(0.0),
            );
            let mut counts = VirtualTilePolishCounts::default();
            let correspondences = virtual_tile_polish_correspondences(
                &left_plane.alignment_image,
                left_coverage,
                &right_plane.alignment_image,
                right_coverage,
                &relation,
                bounds,
                search,
                &mut counts,
            );
            if counts.matched > representative_counts.matched {
                representative_counts = counts;
            }
            if correspondences.len() < STATION_RELATION_MIN_INLIERS {
                continue;
            }
            for model in [
                VirtualTilePolishModel::Similarity,
                VirtualTilePolishModel::Affine,
                VirtualTilePolishModel::Projective,
            ] {
                if restricted_model.is_some_and(|restricted| restricted != model) {
                    continue;
                }
                let Some((homography, inliers)) =
                    model.fit(&correspondences, threshold, left_plane.dimensions())
                else {
                    continue;
                };
                next.push(VirtualTilePolishHypothesis {
                    model,
                    homography,
                    inliers,
                    correspondences: correspondences.clone(),
                });
            }
        }
        if pass_index == 0 {
            hypotheses = next;
        } else {
            // A first-pass lower-DOF model can cover the complete overlap even
            // when its own narrower second-pass search locks onto only one
            // repeated-texture band. Keep both independently measured fits;
            // the shared final gate, not polish iteration order, decides.
            hypotheses.extend(next);
        }
    }

    diagnostic.grid_probe_count = representative_counts.probes;
    diagnostic.measurable_patch_count = representative_counts.textured;
    diagnostic.bidirectional_match_count = representative_counts.matched;
    diagnostic.search_radius_px =
        f64::from(*VIRTUAL_TILE_POLISH_SEARCH_PASSES.last().unwrap_or(&0));
    diagnostic.failure_stage = "polished_fit_failed".to_string();

    let (left_native, right_native) = native_dimensions;
    let mut measured = hypotheses
        .into_iter()
        .filter_map(|hypothesis| {
            let native = normalized_homography(&(to_native * hypothesis.homography * from_native))?;
            let points = hypothesis
                .inliers
                .iter()
                .filter_map(|&index| {
                    let (source, target) = hypothesis.correspondences[index];
                    transformed_point(&to_native, source).zip(transformed_point(&to_native, target))
                })
                .collect::<Vec<_>>();
            (points.len() >= STATION_RELATION_MIN_INLIERS).then_some((
                hypothesis.model,
                native,
                points,
            ))
        })
        .collect::<Vec<_>>();
    measured.sort_by(|left, right| left.0.cmp(&right.0));
    let mut deduplicated = Vec::<(
        VirtualTilePolishModel,
        Matrix3<f64>,
        Vec<(Point2<f64>, Point2<f64>)>,
    )>::new();
    for hypothesis in measured {
        if deduplicated.iter().any(|existing| {
            virtual_tile_hypotheses_are_equivalent(&existing.1, &hypothesis.1, left_native)
        }) {
            continue;
        }
        deduplicated.push(hypothesis);
    }
    if deduplicated.is_empty() {
        return Vec::new();
    }

    let overlap = StationOverlapMeasure::Covered {
        source: left_coverage,
        target: right_coverage,
        factor,
    };
    let representative = deduplicated
        .iter()
        .max_by(|left, right| {
            left.2
                .len()
                .cmp(&right.2.len())
                .then_with(|| right.0.cmp(&left.0))
        })
        .expect("a non-empty hypothesis set has a representative");
    diagnostic.model_fitted = true;
    diagnostic.residual_model = "polished_multi_hypothesis".to_string();
    diagnostic.fitted_inliers = representative.2.len();
    diagnostic.hull_support = station_relation_spatial_support_with(
        &representative.2,
        &representative.1,
        left_native,
        right_native,
        overlap,
    );
    diagnostic.residual_magnitude_px =
        residual_distribution(match representative.1.try_inverse() {
            Some(inverse) => representative
                .2
                .iter()
                .map(|(source, target)| {
                    symmetric_point_error(&representative.1, &inverse, *source, *target)
                })
                .collect(),
            None => Vec::new(),
        });
    diagnostic.failure_stage = "fitted".to_string();

    deduplicated
        .into_iter()
        .map(|(model, homography, points)| {
            let _model_name = model.name();
            MatchInfo {
                homography,
                inliers: points.len(),
                sequence_bridge: false,
                coarse_bridge: false,
                points,
                candidate_points: Vec::new(),
                top_candidate_points: Vec::new(),
                dense_focus_points: Vec::new(),
                foreground_feature_points: Vec::new(),
                canonical_homography: None,
            }
        })
        .collect()
}
