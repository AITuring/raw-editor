//! Bounded native magnification evidence. Only Virtual_Tile callers may attach
//! this evidence; RAW proposals and residual-correction bounds stay unchanged.
use super::virtual_tile_hypothesis::virtual_tile_hypothesis_model_rank;
use super::*;
use crate::panorama_utils::stack_pipeline::report as stack_report;
use nalgebra::{Matrix3, Point2};

/// Split the measured overlap itself into four regions before selecting held
/// points. Whole-image checkerboard cells collapse a narrow overlap into one
/// row and can leave a physically populated region almost absent from holdout.
fn overlap_stratified_holdout(
    points: &[(Point2<f64>, Point2<f64>)],
) -> (
    Vec<(Point2<f64>, Point2<f64>)>,
    Vec<(Point2<f64>, Point2<f64>)>,
) {
    let (mut min_x, mut min_y, mut max_x, mut max_y) = (
        f64::INFINITY,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NEG_INFINITY,
    );
    for (p, _) in points {
        min_x = min_x.min(p.x);
        min_y = min_y.min(p.y);
        max_x = max_x.max(p.x);
        max_y = max_y.max(p.y);
    }
    let (cx, cy) = ((min_x + max_x) * 0.5, (min_y + max_y) * 0.5);
    let mut regions: [Vec<_>; 4] = std::array::from_fn(|_| Vec::new());
    for &pair in points {
        regions[usize::from(pair.0.x >= cx) + 2 * usize::from(pair.0.y >= cy)].push(pair);
    }
    let (mut fit, mut held) = (Vec::new(), Vec::new());
    for region in &mut regions {
        region.sort_by(|a, b| a.0.x.total_cmp(&b.0.x).then(a.0.y.total_cmp(&b.0.y)));
        let desired = region.len().div_ceil(2).max(6).min(region.len());
        // Interleave across each region, then top up sparse regions to six.
        let mut selected = vec![false; region.len()];
        for i in (0..region.len()).step_by(2) {
            selected[i] = true;
        }
        let mut count = selected.iter().filter(|&&v| v).count();
        for i in (1..region.len()).step_by(2) {
            if count >= desired {
                break;
            }
            selected[i] = true;
            count += 1;
        }
        for (&pair, &is_held) in region.iter().zip(&selected) {
            if is_held {
                held.push(pair);
            } else {
                fit.push(pair);
            }
        }
    }
    (fit, held)
}

pub(super) fn measure_virtual_tile_scale_evidence(
    measurements: &mut StationRelationMeasurements,
    transform: &Matrix3<f64>,
    points: &[(Point2<f64>, Point2<f64>)],
    source: (u32, u32),
    target: (u32, u32),
    overlap: Option<StationOverlapMeasure<'_>>,
) {
    // Preserve the accepted near-unit path, including its ranking and reports.
    if (STATION_RELATION_SCALE_RATIO_MIN..=STATION_RELATION_SCALE_RATIO_MAX)
        .contains(&measurements.scale_ratio)
    {
        return;
    }
    let mut evidence = stack_report::StationScaleEvidenceRecord {
        contract_version: 1,
        ..Default::default()
    };
    if !matches!(overlap, Some(StationOverlapMeasure::Covered { .. })) {
        measurements.scale_evidence = Some(evidence);
        return;
    }
    let (fit, held) = overlap_stratified_holdout(points);
    evidence.fit_count = fit.len();
    evidence.holdout_count = held.len();
    let model = match virtual_tile_hypothesis_model_rank(transform) {
        0 => VirtualTilePolishModel::Similarity,
        1 => VirtualTilePolishModel::Affine,
        2 => VirtualTilePolishModel::Projective,
        _ => {
            measurements.scale_evidence = Some(evidence);
            return;
        }
    };
    let refit = model
        .fit(&fit, VIRTUAL_TILE_POLISH_INLIER_THRESHOLD_NATIVE_PX, source)
        .map(|(transform, _)| transform);
    evidence.holdout_median_error_px = refit
        .as_ref()
        .map(|h| median_symmetric_error(h, &held))
        .filter(|v| v.is_finite());
    evidence.holdout_spatial_support = refit.as_ref().and_then(|h| {
        overlap
            .map(|measure| station_relation_spatial_support_with(&held, h, source, target, measure))
    });
    let Some(refit) = refit else {
        measurements.scale_evidence = Some(evidence);
        return;
    };
    let mut bounds = [
        f64::INFINITY,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NEG_INFINITY,
    ];
    for (p, _) in &held {
        bounds[0] = bounds[0].min(p.x);
        bounds[1] = bounds[1].min(p.y);
        bounds[2] = bounds[2].max(p.x);
        bounds[3] = bounds[3].max(p.y);
    }
    let center = Point2::new((bounds[0] + bounds[2]) * 0.5, (bounds[1] + bounds[3]) * 0.5);
    let mut regions: [Vec<_>; 4] = std::array::from_fn(|_| Vec::new());
    for pair in held {
        regions[usize::from(pair.0.x >= center.x) + 2 * usize::from(pair.0.y >= center.y)]
            .push(pair);
    }
    for (i, region) in regions.iter().enumerate() {
        evidence.region_counts[i] = region.len();
        let error = median_symmetric_error(&refit, region);
        evidence.region_median_error_px[i] = error.is_finite().then_some(error);
    }
    let mut min = f64::INFINITY;
    let mut max: f64 = 0.0;
    let mut anisotropy: f64 = 0.0;
    let mut disagreement: f64 = 0.0;
    let mut valid = source.0 > 1 && source.1 > 1;
    for y in 0..9 {
        for x in 0..9 {
            let p = Point2::new(
                f64::from(source.0.saturating_sub(1)) * x as f64 / 8.0,
                f64::from(source.1.saturating_sub(1)) * y as f64 / 8.0,
            );
            let Some((j, r)) = virtual_tile_local_jacobian(transform, p)
                .zip(virtual_tile_local_jacobian(&refit, p))
            else {
                valid = false;
                continue;
            };
            if j.determinant() <= 0.0 || r.determinant() <= 0.0 {
                valid = false;
                continue;
            }
            let scale = j.determinant().sqrt();
            let other = r.determinant().sqrt();
            let sv = j.svd(false, false).singular_values;
            let ratio = sv.max() / sv.min();
            if !scale.is_finite() || !other.is_finite() || !ratio.is_finite() {
                valid = false;
                continue;
            }
            min = min.min(scale);
            max = max.max(scale);
            anisotropy = anisotropy.max(ratio);
            disagreement = disagreement.max((scale / other).max(other / scale) - 1.0);
            evidence.grid_probe_count += 1;
        }
    }
    evidence.positive_jacobians = valid && evidence.grid_probe_count == 81;
    evidence.minimum_scale = min.is_finite().then_some(min);
    evidence.maximum_scale = (max > 0.0).then_some(max);
    evidence.scale_variation = (min > 0.0 && min.is_finite()).then_some(max / min);
    evidence.maximum_anisotropy = (anisotropy > 0.0).then_some(anisotropy);
    evidence.maximum_refit_scale_difference =
        (evidence.grid_probe_count > 0).then_some(disagreement);
    evidence.passes = super::station_relation_gate::bounded_native_scale_is_accepted(&evidence);
    measurements.scale_evidence = Some(evidence);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::panorama_utils::{stack_pipeline::degradation, stitching};
    #[test]
    fn narrow_overlap_holdout_keeps_four_measured_regions() {
        let points = (0..4)
            .flat_map(|region| {
                (0..14).map(move |i| {
                    let x = if region % 2 == 0 { 100.0 } else { 4100.0 } + i as f64 * 11.0;
                    let y = if region < 2 { 100.0 } else { 350.0 } + i as f64 * 2.0;
                    let p = Point2::new(x, y);
                    (p, p)
                })
            })
            .collect::<Vec<_>>();
        let (fit, held) = overlap_stratified_holdout(&points);
        assert!(fit.len() >= 24 && held.len() >= 24);
        let mut regions = [0; 4];
        for (p, _) in held {
            regions[usize::from(p.x >= 2200.0) + 2 * usize::from(p.y >= 250.0)] += 1;
        }
        assert!(regions.iter().all(|&n| n >= 6), "{regions:?}");
    }
    fn measured(scale: f64) -> StationRelationMeasurements {
        StationRelationMeasurements {
            scale_evidence: None,
            inliers: 576,
            median_error_px: 0.0,
            scale_ratio: scale,
            spatial_support: 0.6,
            low_frequency_mean_relative_difference: 0.0,
            edge_strength_ratio: 1.0,
            median_edge_orientation_difference_degrees: 0.0,
            preserves_convex_orientation: true,
        }
    }
    fn points(h: &Matrix3<f64>) -> Vec<(Point2<f64>, Point2<f64>)> {
        (0..24)
            .flat_map(|y| {
                (0..24).map(move |x| {
                    let p = Point2::new(50.0 + x as f64 * 30.0, 50.0 + y as f64 * 30.0);
                    (p, transformed_point(h, p).unwrap())
                })
            })
            .collect()
    }
    fn evaluate(
        h: &Matrix3<f64>,
        pairs: &[(Point2<f64>, Point2<f64>)],
    ) -> StationRelationMeasurements {
        let coverage = stitching::CoverageMask::from_bytes(250, 250, vec![1; 250 * 250]).unwrap();
        let mut m = measured(station_relation_overlap_scale(h, pairs).unwrap_or(f64::NAN));
        measure_virtual_tile_scale_evidence(
            &mut m,
            h,
            pairs,
            (1000, 1000),
            (1000, 1000),
            Some(StationOverlapMeasure::Covered {
                source: &coverage,
                target: &coverage,
                factor: 4,
            }),
        );
        m
    }
    #[test]
    fn bounded_scale_accepts_native_magnification_in_both_directions() {
        for scale in [0.9, 1.12] {
            let h = Matrix3::new(scale, 0.0, 30.0, 0.0, scale, 20.0, 0.0, 0.0, 1.0);
            let p = points(&h);
            for (h, p) in [
                (h, p.clone()),
                (
                    h.try_inverse().unwrap(),
                    p.iter().map(|&(s, t)| (t, s)).collect(),
                ),
            ] {
                let m = evaluate(&h, &p);
                assert!(station_relation_rejection_reasons(&m).is_empty(), "{:?}", m);
                let e = m.scale_evidence.unwrap();
                assert!(e.passes && e.grid_probe_count == 81);
                assert!(e.region_counts.iter().all(|&count| count >= 6));
            }
        }
    }
    #[test]
    fn bounded_scale_preserves_legacy_and_raw_contract() {
        for scale in [0.95, 1.0, 1.05] {
            let h = Matrix3::new(scale, 0.0, 0.0, 0.0, scale, 0.0, 0.0, 0.0, 1.0);
            let mut m = measured(scale);
            measure_virtual_tile_scale_evidence(
                &mut m,
                &h,
                &points(&h),
                (1000, 1000),
                (1000, 1000),
                None,
            );
            assert!(m.scale_evidence.is_none());
            assert!(station_relation_rejection_reasons(&m).is_empty());
        }
        let h = Matrix3::new(0.9, 0.0, 0.0, 0.0, 0.9, 0.0, 0.0, 0.0, 1.0);
        let mut m = measured(0.9);
        measure_virtual_tile_scale_evidence(
            &mut m,
            &h,
            &points(&h),
            (1000, 1000),
            (1000, 1000),
            Some(StationOverlapMeasure::Rectangles),
        );
        assert!(
            station_relation_rejection_reasons(&m)
                .contains(&degradation::STATION_RELATION_SCALE_OUT_OF_RANGE)
        );
    }
    #[test]
    fn bounded_scale_rejects_envelope_anisotropy_folding_and_missing_support() {
        for h in [
            Matrix3::new(0.79, 0.0, 0.0, 0.0, 0.79, 0.0, 0.0, 0.0, 1.0),
            Matrix3::new(1.26, 0.0, 0.0, 0.0, 1.26, 0.0, 0.0, 0.0, 1.0),
            Matrix3::new(1.3, 0.0, 0.0, 0.0, 0.9, 0.0, 0.0, 0.0, 1.0),
            Matrix3::new(-0.9, 0.0, 800.0, 0.0, 0.9, 0.0, 0.0, 0.0, 1.0),
        ] {
            assert!(!evaluate(&h, &points(&h)).scale_evidence.unwrap().passes);
        }
        let h = Matrix3::new(0.9, 0.0, 0.0, 0.0, 0.9, 0.0, 0.0, 0.0, 1.0);
        let p = points(&h);
        assert!(!evaluate(&h, &p[..20]).scale_evidence.unwrap().passes);
        let line = p
            .iter()
            .copied()
            .filter(|(p, _)| p.y == 50.0)
            .collect::<Vec<_>>();
        assert!(!evaluate(&h, &line).scale_evidence.unwrap().passes);
    }
    #[test]
    fn bounded_scale_rejects_bad_holdout_and_never_waives_photometric_gate() {
        let h = Matrix3::new(0.9, 0.0, 0.0, 0.0, 0.9, 0.0, 0.0, 0.0, 1.0);
        let mut p = points(&h);
        for (s, t) in &mut p {
            if ((s.x / (1000.0 / 12.0)).floor() as i64 + (s.y / (1000.0 / 12.0)).floor() as i64)
                .rem_euclid(2)
                == 1
            {
                t.x += 20.0;
            }
        }
        assert!(!evaluate(&h, &p).scale_evidence.unwrap().passes);
        let mut m = evaluate(&h, &points(&h));
        m.low_frequency_mean_relative_difference = 0.21;
        assert_eq!(
            station_relation_rejection_reasons(&m),
            vec![degradation::STATION_RELATION_PHOTOMETRIC_MISMATCH]
        );
        m.scale_evidence
            .as_mut()
            .unwrap()
            .maximum_refit_scale_difference = Some(f64::NAN);
        assert!(
            station_relation_rejection_reasons(&m)
                .contains(&degradation::STATION_RELATION_SCALE_OUT_OF_RANGE)
        );
    }
}
