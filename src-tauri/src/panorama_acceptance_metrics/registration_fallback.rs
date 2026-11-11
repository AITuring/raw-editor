//! Test-only partial-output/full-reference registration. Reference pixels never
//! enter rendering. Keep the original matcher first; search a bounded physical
//! scale range only after that path fails.
use super::*;
use crate::panorama_utils::shared::Feature;

type Pair = (Point2<f64>, Point2<f64>);

pub(super) fn partial_reference_registration(
    output: &GrayImage,
    reference_features: &[Feature],
    full_reference: &GrayImage,
    reference_reg_dimensions: (u32, u32),
    brief_pairs: &[(Point2<i32>, Point2<i32>)],
) -> Result<(Matrix3<f64>, usize, usize, Value), String> {
    let mut best: Option<(Matrix3<f64>, usize, usize, Value)> = None;
    let mut attempts = Vec::new();
    for factor in [0.75, 0.5, 0.375, 0.25, 0.1875] {
        let w = (f64::from(output.width()) * factor).round() as u32;
        let h = (f64::from(output.height()) * factor).round() as u32;
        if w.min(h) < 96 {
            continue;
        }
        let scaled = image::imageops::resize(output, w, h, FilterType::Triangle);
        let features = processing::find_features_multiscale(
            &scaled,
            brief_pairs,
            REGISTRATION_MAX_FEATURES,
            &REGISTRATION_SCALES,
        );
        let matches = processing::match_features_with_ratio(
            &features,
            reference_features,
            REGISTRATION_RATIO,
        );
        let points = matches
            .iter()
            .map(|candidate| {
                let p = features[candidate.index1].keypoint;
                let q = reference_features[candidate.index2].keypoint;
                (
                    Point2::new(
                        f64::from(p.x) * f64::from(output.width()) / f64::from(w),
                        f64::from(p.y) * f64::from(output.height()) / f64::from(h),
                    ),
                    Point2::new(f64::from(q.x), f64::from(q.y)),
                )
            })
            .collect::<Vec<_>>();
        let (fit, held): (Vec<_>, Vec<_>) = points.iter().copied().partition(|(p, _)| {
            ((p.x / 64.0).floor() as i64 + (p.y / 64.0).floor() as i64) % 2 == 0
        });
        let Some((ransac, indices)) =
            processing::find_homography_ransac_points_stable_with_min_inliers(
                &fit,
                REGISTRATION_INLIER_PX,
                REGISTRATION_MIN_INLIERS,
            )
        else {
            attempts.push(
                json!({"factor":factor,"matches":points.len(),"reason":"fit_inliers_below_40"}),
            );
            continue;
        };
        let inliers = indices.iter().map(|&i| fit[i]).collect::<Vec<_>>();
        let model = processing::compute_homography(&inliers).unwrap_or(ransac);
        let Some(mut evidence) = validate_holdout(&model, &held, output.dimensions()) else {
            attempts.push(json!({"factor":factor,"matches":points.len(),"fit_inliers":indices.len(),"reason":"independent_spatial_holdout_failed","holdout":holdout_diagnostics(&model,&held,output.dimensions())}));
            continue;
        };
        evidence["method"] = json!("bounded_partial_reference_scale_search_v1");
        evidence["factor"] = json!(factor);
        evidence["fit_matches"] = json!(fit.len());
        evidence["fit_inliers"] = json!(indices.len());
        let count = indices.len() + evidence["holdout_inliers"].as_u64().unwrap() as usize;
        if best.as_ref().is_none_or(|value| {
            evidence["validated_output_fraction"]
                .as_f64()
                .unwrap_or(0.0)
                > value.3["validated_output_fraction"].as_f64().unwrap_or(0.0)
        }) {
            best = Some((model, points.len(), count, evidence));
        }
    }
    if best.is_none() {
        let output_features = processing::find_features_multiscale(
            output,
            brief_pairs,
            REGISTRATION_MAX_FEATURES,
            &REGISTRATION_SCALES,
        );
        let (rw, rh) = full_reference.dimensions();
        let aspect = f64::from(output.width()) / f64::from(output.height());
        let (cw, ch) = if f64::from(rw) / f64::from(rh) > aspect {
            (((f64::from(rh) * aspect).round() as u32).clamp(1, rw), rh)
        } else {
            (rw, ((f64::from(rw) / aspect).round() as u32).clamp(1, rh))
        };
        let span = (rw - cw).max(rh - ch);
        let step = (cw.min(ch) / 2).max(1);
        let windows = (span.div_ceil(step) + 1).clamp(1, 12);
        for index in 0..windows {
            let offset = if windows == 1 {
                0
            } else {
                u64::from(span) * u64::from(index) / u64::from(windows - 1)
            } as u32;
            let (left, top) = if rw > cw { (offset, 0) } else { (0, offset) };
            let crop = image::imageops::crop_imm(full_reference, left, top, cw, ch).to_image();
            let (w, h) = fit_long_side(cw, ch, REGISTRATION_LONG_SIDE);
            let window = image::imageops::resize(&crop, w, h, FilterType::Triangle);
            drop(crop);
            let features = processing::find_features_multiscale(
                &window,
                brief_pairs,
                REGISTRATION_MAX_FEATURES,
                &REGISTRATION_SCALES,
            );
            let matches = processing::match_features_with_ratio(
                &output_features,
                &features,
                REGISTRATION_RATIO,
            );
            // Fit and validate in the original whole-reference 2048 lattice.
            // A 3px window residual would silently impose a different physical
            // tolerance for each crop size.
            let window_map = reference_window_map(
                (rw, rh),
                reference_reg_dimensions,
                (left, top),
                (cw, ch),
                (w, h),
            );
            let points = matches
                .iter()
                .map(|candidate| {
                    let p = output_features[candidate.index1].keypoint;
                    let q = features[candidate.index2].keypoint;
                    (
                        Point2::new(f64::from(p.x), f64::from(p.y)),
                        project(&window_map, Point2::new(f64::from(q.x), f64::from(q.y)))
                            .expect("finite window scale"),
                    )
                })
                .collect::<Vec<_>>();
            let (fit, held): (Vec<_>, Vec<_>) = points.iter().copied().partition(|(p, _)| {
                ((p.x / 64.0).floor() as i64 + (p.y / 64.0).floor() as i64) % 2 == 0
            });
            let Some((ransac, indices)) =
                processing::find_homography_ransac_points_stable_with_min_inliers(
                    &fit,
                    REGISTRATION_INLIER_PX,
                    REGISTRATION_MIN_INLIERS,
                )
            else {
                attempts.push(json!({"window":[left,top,cw,ch],"matches":points.len(),"reason":"fit_inliers_below_40"}));
                continue;
            };
            let inliers = indices.iter().map(|&i| fit[i]).collect::<Vec<_>>();
            let model = processing::compute_homography(&inliers).unwrap_or(ransac);
            let Some(mut evidence) = validate_holdout(&model, &held, output.dimensions()) else {
                attempts.push(json!({"window":[left,top,cw,ch],"matches":points.len(),"fit_inliers":indices.len(),"reason":"independent_spatial_holdout_failed","holdout":holdout_diagnostics(&model,&held,output.dimensions())}));
                continue;
            };
            evidence["method"] = json!("bounded_partial_reference_windows_v1");
            evidence["reference_window_native"] = json!([left, top, cw, ch]);
            evidence["reference_window_registration"] = json!([w, h]);
            evidence["error_coordinate_space"] = json!("whole_reference_2048_registration_pixels");
            evidence["fit_matches"] = json!(fit.len());
            evidence["fit_inliers"] = json!(indices.len());
            evidence["window_count_limit"] = json!(12);
            let count = indices.len() + evidence["holdout_inliers"].as_u64().unwrap() as usize;
            if best.as_ref().is_none_or(|value| {
                evidence["validated_output_fraction"]
                    .as_f64()
                    .unwrap_or(0.0)
                    > value.3["validated_output_fraction"].as_f64().unwrap_or(0.0)
            }) {
                best = Some((model, points.len(), count, evidence));
            }
        }
    }
    if let Some(mut result) = best {
        result.3["rejected_attempts"] = json!(attempts);
        Ok(result)
    } else {
        Err(format!(
            "bounded partial-reference registration failed: {}",
            json!(attempts)
        ))
    }
}

fn reference_window_map(
    native: (u32, u32),
    registration: (u32, u32),
    origin: (u32, u32),
    crop: (u32, u32),
    window: (u32, u32),
) -> Matrix3<f64> {
    let shift = Matrix3::new(
        1.0,
        0.0,
        f64::from(origin.0),
        0.0,
        1.0,
        f64::from(origin.1),
        0.0,
        0.0,
        1.0,
    );
    scale_matrix(
        f64::from(registration.0) / f64::from(native.0),
        f64::from(registration.1) / f64::from(native.1),
    ) * shift
        * scale_matrix(
            f64::from(crop.0) / f64::from(window.0),
            f64::from(crop.1) / f64::from(window.1),
        )
}

fn holdout_diagnostics(model: &Matrix3<f64>, held: &[Pair], dimensions: (u32, u32)) -> Value {
    let mut errors = Vec::new();
    let mut regions = [0usize; 4];
    let mut region_errors: [Vec<f64>; 4] = Default::default();
    for &(p, q) in held {
        let error = project(model, p)
            .map(|v| (v - q).norm())
            .unwrap_or(f64::INFINITY);
        errors.push(error);
        let region = usize::from(p.x >= f64::from(dimensions.0) * 0.5)
            + 2 * usize::from(p.y >= f64::from(dimensions.1) * 0.5);
        if error <= REGISTRATION_INLIER_PX {
            regions[region] += 1;
            region_errors[region].push(error);
        }
    }
    json!({"count":held.len(),"within_3px":regions.iter().sum::<usize>(),"quadrants":regions,
        "median_all":percentile(&errors,0.5),"region_median_inlier":region_errors.iter().map(|v|percentile(v,0.5)).collect::<Vec<_>>()})
}

fn validate_holdout(model: &Matrix3<f64>, held: &[Pair], dimensions: (u32, u32)) -> Option<Value> {
    if !model.iter().all(|x| x.is_finite()) || held.len() < 20 {
        return None;
    }
    let mut regions = [0usize; 4];
    let mut supported = Vec::new();
    let mut errors = Vec::new();
    let mut total_errors = Vec::new();
    for &(p, q) in held {
        let error = (project(model, p)? - q).norm();
        total_errors.push(error);
        if error <= REGISTRATION_INLIER_PX {
            supported.push(p);
            errors.push(error);
        }
    }
    if errors.len() < 20 || errors.len() * 2 < held.len() {
        return None;
    }
    // A partial reference can validate only part of the delivered canvas.
    // Establish a broad independently supported region, then forbid metric
    // extrapolation outside its hull. Empty frame/absent reference cannot
    // manufacture either inliers or a global registration claim.
    let hull = convex_hull(&supported);
    let area = polygon_area(&hull);
    let fraction = area / (f64::from(dimensions.0) * f64::from(dimensions.1));
    let min_x = supported.iter().map(|p| p.x).fold(f64::INFINITY, f64::min);
    let max_x = supported
        .iter()
        .map(|p| p.x)
        .fold(f64::NEG_INFINITY, f64::max);
    let min_y = supported.iter().map(|p| p.y).fold(f64::INFINITY, f64::min);
    let max_y = supported
        .iter()
        .map(|p| p.y)
        .fold(f64::NEG_INFINITY, f64::max);
    if fraction < 0.20
        || max_x - min_x < 0.35 * f64::from(dimensions.0)
        || max_y - min_y < 0.35 * f64::from(dimensions.1)
    {
        return None;
    }
    for p in &supported {
        let region = usize::from(p.x >= (min_x + max_x) * 0.5)
            + 2 * usize::from(p.y >= (min_y + max_y) * 0.5);
        regions[region] += 1;
    }
    if regions.iter().any(|&n| n < 3) {
        return None;
    }
    // A fitted reflection/pole is not an artwork registration.
    for y in [0.0, 0.5, 1.0] {
        for x in [0.0, 0.5, 1.0] {
            let p = Point2::new(x * f64::from(dimensions.0), y * f64::from(dimensions.1));
            let q = project(model, p)?;
            let dx = project(model, p + nalgebra::Vector2::new(1.0, 0.0))? - q;
            let dy = project(model, p + nalgebra::Vector2::new(0.0, 1.0))? - q;
            if dx.x * dy.y - dx.y * dy.x <= 1e-8 {
                return None;
            }
        }
    }
    Some(
        json!({"holdout_matches":held.len(),"holdout_inliers":errors.len(),
        "holdout_support":errors.len() as f64 / held.len() as f64,
        "holdout_all_median_px":percentile(&total_errors,0.5),
        "holdout_inlier_p95_px":percentile(&errors,0.95),"quadrant_inliers":regions,
        "inlier_limit_px":REGISTRATION_INLIER_PX,"minimum_fit_inliers":REGISTRATION_MIN_INLIERS,
        "validated_output_fraction":fraction,
        "validated_output_polygon_registration":hull.iter().map(|p|[p.x,p.y]).collect::<Vec<_>>(),
        "minimum_validated_output_fraction":0.20,"minimum_axis_span_fraction":0.35}),
    )
}

fn cross(a: Point2<f64>, b: Point2<f64>, c: Point2<f64>) -> f64 {
    (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)
}
fn convex_hull(points: &[Point2<f64>]) -> Vec<Point2<f64>> {
    let mut points = points.to_vec();
    points.sort_by(|a, b| a.x.total_cmp(&b.x).then(a.y.total_cmp(&b.y)));
    points.dedup();
    if points.len() < 3 {
        return vec![];
    }
    let mut lower = Vec::new();
    let mut upper = Vec::new();
    for &p in &points {
        while lower.len() >= 2 && cross(lower[lower.len() - 2], lower[lower.len() - 1], p) <= 0.0 {
            lower.pop();
        }
        lower.push(p);
    }
    for &p in points.iter().rev() {
        while upper.len() >= 2 && cross(upper[upper.len() - 2], upper[upper.len() - 1], p) <= 0.0 {
            upper.pop();
        }
        upper.push(p);
    }
    lower.pop();
    upper.pop();
    lower.extend(upper);
    lower
}
fn polygon_area(points: &[Point2<f64>]) -> f64 {
    if points.len() < 3 {
        return 0.0;
    }
    points
        .iter()
        .zip(points.iter().cycle().skip(1))
        .map(|(a, b)| a.x * b.y - a.y * b.x)
        .sum::<f64>()
        .abs()
        * 0.5
}
pub(super) fn region_contains(points: &[Point2<f64>], point: Point2<f64>) -> bool {
    points.len() >= 3
        && point.x.is_finite()
        && point.y.is_finite()
        && points
            .iter()
            .zip(points.iter().cycle().skip(1))
            .all(|(&a, &b)| cross(a, b, point) >= -1e-8)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn partial_reference_metrics_cannot_extrapolate_outside_supported_hull() {
        let hull = convex_hull(&[
            Point2::new(0.0, 0.0),
            Point2::new(100.0, 0.0),
            Point2::new(100.0, 100.0),
            Point2::new(0.0, 100.0),
            Point2::new(50.0, 50.0),
        ]);
        assert_eq!(polygon_area(&hull), 10_000.0);
        assert!(region_contains(&hull, Point2::new(50.0, 50.0)));
        assert!(!region_contains(&hull, Point2::new(101.0, 50.0)));
        assert!(!region_contains(&hull, Point2::new(f64::NAN, 50.0)));
    }

    #[test]
    fn windows_keep_the_original_reference_error_units() {
        let cropped = reference_window_map(
            (8000, 2000),
            (2048, 512),
            (6000, 0),
            (2000, 2000),
            (2048, 2048),
        );
        let whole =
            reference_window_map((8000, 2000), (2048, 512), (0, 0), (8000, 2000), (2048, 512));
        let a = project(&cropped, Point2::new(512.0, 512.0)).unwrap();
        let b = project(&whole, Point2::new(1664.0, 128.0)).unwrap();
        assert!((a - b).norm() < 1e-10);
        let error = project(&cropped, Point2::new(524.0, 512.0)).unwrap() - a;
        assert!((error.norm() - REGISTRATION_INLIER_PX).abs() < 1e-10);
    }

    #[test]
    fn holdout_rejects_wrong_models_reflection_and_single_region() {
        let pairs = (0..8)
            .flat_map(|y| {
                (0..8).map(move |x| {
                    let p = Point2::new(20.0 + x as f64 * 25.0, 20.0 + y as f64 * 25.0);
                    (p, p)
                })
            })
            .collect::<Vec<_>>();
        assert!(validate_holdout(&Matrix3::identity(), &pairs, (240, 240)).is_some());
        let wrong = Matrix3::new(1.0, 0.0, 10.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);
        assert!(validate_holdout(&wrong, &pairs, (240, 240)).is_none());
        assert!(validate_holdout(&Matrix3::identity(), &pairs, (1000, 1000)).is_none());
        let reflection = Matrix3::new(-1.0, 0.0, 240.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);
        let reflected = pairs
            .iter()
            .map(|(p, _)| (*p, project(&reflection, *p).unwrap()))
            .collect::<Vec<_>>();
        assert!(validate_holdout(&reflection, &reflected, (240, 240)).is_none());
    }
}
