use super::*;

#[test]
fn density_gauge_uses_a_feasible_common_pivot_without_relaxing_corner_limits() {
    let initial = Matrix3::new(0.96, 0.0, 0.0, 0.0, 0.96, 0.0, 0.0, 0.0, 1.0);
    let pose = Matrix3::new(0.96, 0.0, 100.0, 0.0, 0.96, 0.0, 0.0, 0.0, 1.0);
    let result = native_density_gauge(
        Decomposition {
            poses: vec![pose],
            gauge: Matrix3::identity(),
            maximum_corner_motion: 100.0,
        },
        &[(1024, 1024)],
        &[initial],
        &[112.0],
    )
    .unwrap();
    assert!(result.poses[0].determinant().sqrt() > 1.0 - 1e-7);
    assert!(result.maximum_corner_motion <= 112.0 + 1e-7);
    for p in corners((1024, 1024)) {
        assert!(
            (project(&result.poses[0], p).unwrap() - project(&initial, p).unwrap()).norm()
                <= 112.0 + 1e-7
        );
    }
}

#[test]
fn density_gauge_recovers_pixels_without_exceeding_camera_budgets() {
    let pose = Matrix3::new(0.96, 0.0, 0.0, 0.0, 0.96, 0.0, 0.0, 0.0, 1.0);
    let run = |limit| {
        native_density_gauge(
            Decomposition {
                poses: vec![pose],
                gauge: Matrix3::identity(),
                maximum_corner_motion: 0.0,
            },
            &[(1024, 1024)],
            &[pose],
            &[limit],
        )
        .unwrap()
    };
    let recovered = run(32.0);
    assert!(recovered.poses[0].determinant().sqrt() >= 0.98 - 1e-7);
    assert!(recovered.maximum_corner_motion <= 32.0);
    let limited = run(2.0);
    assert!(limited.poses[0].determinant().sqrt() > 0.96);
    assert!(limited.poses[0].determinant().sqrt() < 0.98);
    assert!(limited.maximum_corner_motion <= 2.0);
    for point in corners((1024, 1024)) {
        assert!(
            (project(&limited.poses[0], point).unwrap() - project(&pose, point).unwrap()).norm()
                <= 2.0
        );
    }
}
#[test]
fn corner_budget_can_move_into_a_bounded_local_field_but_cannot_be_forced() {
    let proposal = [Matrix3::identity(); 2];
    let initial = [
        Matrix3::identity(),
        Matrix3::new(1.06, 0.0, 0.0, 0.0, 1.06, 0.0, 0.0, 0.0, 1.0),
    ];
    let points = (128..896)
        .step_by(64)
        .flat_map(|x| {
            (128..896).step_by(64).map(move |y| {
                let p = Point2::new(x as f64, y as f64);
                (p, p)
            })
        })
        .collect::<Vec<_>>();
    let relations = [Relation {
        left: 0,
        right: 1,
        points,
    }];
    let fit = fit(&proposal, &[(1024, 1024); 2], &relations, 0).unwrap();
    let verified = (0..2)
        .map(|_| {
            relations[0]
                .points
                .iter()
                .map(|&(point, _)| VerifiedSource {
                    point,
                    round_trip_error_px: 0.2,
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let decomposed = decompose(
        &proposal,
        &fit.fields,
        &[(1024, 1024); 2],
        &initial,
        &[24.0; 2],
    );
    assert!(
        decomposed.is_some(),
        "the bounded camera/local decomposition is feasible"
    );
    let candidate = materialize(
        &proposal,
        &fit,
        &[(1024, 1024); 2],
        &initial,
        &[24.0; 2],
        &relations,
        &verified,
    )
    .unwrap();
    assert!(candidate.maximum_corner_motion <= 24.0);
    assert!(!candidate.residual.identity());
    assert!(
        candidate
            .residual
            .regions
            .iter()
            .all(|r| r.max_node_displacement_px <= 32.0)
    );
    assert!(
        materialize(
            &proposal,
            &fit,
            &[(1024, 1024); 2],
            &initial,
            &[1.0; 2],
            &relations,
            &verified
        )
        .is_none(),
        "an infeasible budget must still reject the model"
    );
}
#[test]
fn guarded_model_requires_native_evidence_and_rechecks_actual_sampling() {
    let poses = [Matrix3::identity(); 2];
    let mut points = Vec::new();
    for x in (32..992).step_by(32) {
        for y in (32..992).step_by(32) {
            let a = Point2::new(x as f64, y as f64);
            points.push((a, Point2::new(a.x + 12.0 * (a.y / 150.0).sin(), a.y)));
        }
    }
    let relation = Relation {
        left: 0,
        right: 1,
        points,
    };
    let relations = [relation];
    let fields = fit(&poses, &[(1024, 1024); 2], &relations, 0).unwrap();
    assert!(
        materialize(
            &poses,
            &fields,
            &[(1024, 1024); 2],
            &poses,
            &[256.0; 2],
            &relations,
            &[Vec::new(), Vec::new()]
        )
        .is_none()
    );
    let verified = (0..2)
        .map(|station| {
            relations[0]
                .points
                .iter()
                .map(|&(a, b)| VerifiedSource {
                    point: if station == 0 { a } else { b },
                    round_trip_error_px: 0.4,
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let bounded = materialize(
        &poses,
        &fields,
        &[(1024, 1024); 2],
        &poses,
        &[256.0; 2],
        &relations,
        &verified,
    )
    .unwrap();
    assert!(!bounded.residual.identity());
    assert!(bounded.validation[0].p95.is_finite());
    // Boundary fading is a real change to the fitted mapping. Its errors
    // must be measured instead of copying the unguarded fit statistics.
    assert!((bounded.validation[0].p95 - fields.validation[0].p95).abs() > 0.01);
    for region in &bounded.residual.regions {
        assert!(region.max_node_displacement_px <= 32.0);
        assert!(region.max_neighbour_delta_px <= 8.0 + 1e-8);
        assert_eq!(
            region.displacement_at(
                region.world.left,
                region.world.top + region.world.height / 2.0
            ),
            [0.0; 2]
        );
    }
    let invalid = verified
        .iter()
        .map(|values| {
            values
                .iter()
                .map(|v| VerifiedSource {
                    round_trip_error_px: 1.1,
                    ..*v
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    assert!(
        materialize(
            &poses,
            &fields,
            &[(1024, 1024); 2],
            &poses,
            &[256.0; 2],
            &relations,
            &invalid
        )
        .is_none()
    );
}
#[test]
fn projective_component_recovers_camera_motion_without_moving_the_reference() {
    let initial = vec![
        Matrix3::identity(),
        Matrix3::new(
            1.003, 0.002, 317.0, -0.001, 0.997, 9.0, 0.000002, -0.000001, 1.0,
        ),
        Matrix3::new(
            0.998, -0.003, 583.0, 0.001, 1.003, -7.0, -0.000001, 0.000002, 1.0,
        ),
    ];
    let mut relations = Vec::new();
    for (left, right) in [(0, 1), (1, 2), (0, 2)] {
        let mut points = Vec::new();
        for y in (80..920).step_by(40) {
            for x in (640..960).step_by(40) {
                points.push((
                    Point2::new(x as f64 - left as f64 * 300.0, y as f64),
                    Point2::new(x as f64 - right as f64 * 300.0, y as f64),
                ));
            }
        }
        relations.push(Relation {
            left,
            right,
            points,
        });
    }
    let result = projective_proposal(&initial, &relations, 0, 1024.0).unwrap();
    assert_eq!(result[0], initial[0]);
    for relation in &relations {
        for &(a, b) in &relation.points {
            let error = (project(&result[relation.left], a).unwrap()
                - project(&result[relation.right], b).unwrap())
            .norm();
            assert!(error < 0.05, "camera motion did not converge: {error}");
        }
    }
    let fields = fit(&result, &[(1024, 1024); 3], &relations, 0).unwrap();
    let decomposed = decompose(
        &result,
        &fields.fields,
        &[(1024, 1024); 3],
        &initial,
        &[256.0; 3],
    )
    .unwrap();
    assert!(decomposed.maximum_corner_motion < 40.0);
    assert!(decomposed.poses.iter().all(|h| h.try_inverse().is_some()));
}
#[test]
#[ignore = "requires saved native station correspondences and a diagnostic pose set"]
fn real_joint_station_field_diagnostics_from_env() {
    let input = std::env::var("RAW_EDITOR_JOINT_WARP_INPUT").unwrap();
    let output = std::env::var("RAW_EDITOR_JOINT_WARP_OUTPUT").unwrap();
    let data: serde_json::Value = serde_json::from_slice(&std::fs::read(input).unwrap()).unwrap();
    let mut poses = data["poses"]
        .as_array()
        .unwrap()
        .iter()
        .map(|p| Matrix3::from_fn(|row, column| p[row][column].as_f64().unwrap()))
        .collect::<Vec<_>>();
    let dimensions = data["dimensions"]
        .as_array()
        .unwrap()
        .iter()
        .map(|d| (d[0].as_u64().unwrap() as u32, d[1].as_u64().unwrap() as u32))
        .collect::<Vec<_>>();
    let relations = data["records"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| Relation {
            left: r["left"].as_u64().unwrap() as usize,
            right: r["right"].as_u64().unwrap() as usize,
            points: r["points"]
                .as_array()
                .unwrap()
                .iter()
                .map(|p| {
                    (
                        Point2::new(p[0].as_f64().unwrap(), p[1].as_f64().unwrap()),
                        Point2::new(p[2].as_f64().unwrap(), p[3].as_f64().unwrap()),
                    )
                })
                .collect(),
        })
        .collect::<Vec<_>>();
    let initial = poses.clone();
    if let Some(proposal) = data["proposal_poses"].as_array() {
        poses = proposal
            .iter()
            .map(|p| Matrix3::from_fn(|row, column| p[row][column].as_f64().unwrap()))
            .collect();
    } else if std::env::var_os("RAW_EDITOR_JOINT_WARP_OPTIMIZE").is_some() {
        poses = projective_proposal(&poses, &relations, 0, 9000.0).unwrap();
    }
    let result = fit(&poses, &dimensions, &relations, 0).unwrap();
    let bounded = if let Some(limits) = data["limits"].as_array() {
        let limits = limits
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect::<Vec<_>>();
        let mut verified = vec![Vec::new(); poses.len()];
        for record in data["records"].as_array().unwrap() {
            let (left, right) = (
                record["left"].as_u64().unwrap() as usize,
                record["right"].as_u64().unwrap() as usize,
            );
            if let Some(points) = record["native_verified_points"].as_array() {
                for point in points {
                    let error = point[4].as_f64().unwrap();
                    verified[left].push(VerifiedSource {
                        point: Point2::new(point[0].as_f64().unwrap(), point[1].as_f64().unwrap()),
                        round_trip_error_px: error,
                    });
                    verified[right].push(VerifiedSource {
                        point: Point2::new(point[2].as_f64().unwrap(), point[3].as_f64().unwrap()),
                        round_trip_error_px: error,
                    });
                }
            }
        }
        materialize(
            &poses,
            &result,
            &dimensions,
            &initial,
            &limits,
            &relations,
            &verified,
        )
        .map(|bounded| {
            serde_json::json!({
                "max_corner": bounded.maximum_corner_motion,
                "regions": bounded.residual.report_records(),
                "validation": bounded.validation.iter().map(|v| serde_json::json!({
                    "left":v.left,"right":v.right,"count":v.count,"median":v.median,"p95":v.p95,
                })).collect::<Vec<_>>(),
            })
        })
    } else {
        None
    };
    let fields = result.fields.iter().map(|f| serde_json::json!({
        "origin":[f.origin.x,f.origin.y],"columns":f.columns,"rows":f.rows,"values":f.values,
    })).collect::<Vec<_>>();
    let validation = result
        .validation
        .iter()
        .map(|v| {
            serde_json::json!({
                "left":v.left,"right":v.right,"count":v.count,"median":v.median,"p95":v.p95,
            })
        })
        .collect::<Vec<_>>();
    let max_node = result
        .fields
        .iter()
        .flat_map(|f| &f.values)
        .map(|v| v[0].hypot(v[1]))
        .fold(0.0, f64::max);
    println!(
        "Joint station diagnostic: {} relations, max unbounded node {max_node:.3}px",
        relations.len()
    );
    for v in &result.validation {
        println!(
            "PAIR {} {} count={} median={:.3} p95={:.3}",
            v.left, v.right, v.count, v.median, v.p95
        );
    }
    std::fs::write(
        output,
        serde_json::to_vec(&serde_json::json!({
            "fields":fields,"validation":validation,"max_node":max_node,"bounded":bounded,
            "poses":poses.iter().map(|h| (0..3).map(|r| (0..3).map(|c| h[(r,c)]).collect::<Vec<_>>()).collect::<Vec<_>>()).collect::<Vec<_>>(),
        }))
        .unwrap(),
    )
    .unwrap();
}
#[test]
fn validation_observations_cannot_be_fitted_away() {
    let poses = [Matrix3::identity(); 2];
    let mut points = Vec::new();
    for x in (64..960).step_by(32) {
        for y in (64..960).step_by(32) {
            let point = Point2::new(x as f64, y as f64);
            let offset = if points.len() % 5 == 0 { 15.0 } else { 0.0 };
            points.push((point, Point2::new(point.x + offset, point.y)));
        }
    }
    let result = fit(
        &poses,
        &[(1024, 1024); 2],
        &[Relation {
            left: 0,
            right: 1,
            points,
        }],
        0,
    )
    .unwrap();
    assert!(
        result.validation[0].p95 >= 14.9,
        "held-out errors were hidden by fitting: {:?}",
        result.validation
    );
    assert!(
        result
            .fields
            .iter()
            .all(|f| f.values.iter().all(|v| *v == [0.0; 2]))
    );
}
#[test]
fn joint_fit_uses_validation_points_and_recovers_a_smooth_nonrigid_overlap() {
    let poses = [Matrix3::identity(); 2];
    let mut points = Vec::new();
    for y in (64..960).step_by(32) {
        for x in (64..960).step_by(32) {
            let a = Point2::new(x as f64, y as f64);
            let delta = 2.0 * (a.y / 300.0).sin();
            points.push((a, Point2::new(a.x + delta, a.y)));
        }
    }
    let fit = fit(
        &poses,
        &[(1024, 1024); 2],
        &[Relation {
            left: 0,
            right: 1,
            points,
        }],
        0,
    )
    .unwrap();
    assert!(fit.validation[0].count > 100);
    assert!(
        fit.validation[0].p95 < 0.5,
        "validation: {:?}",
        fit.validation
    );
    assert!(fit.fields[0].values.iter().all(|v| *v == [0.0; 2]));
    assert!(fit.fields[1].at(Point2::new(512.0, 256.0))[0].abs() > 1.0);
}
