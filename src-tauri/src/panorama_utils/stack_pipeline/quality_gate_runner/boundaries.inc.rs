
/// Measure only accepted inter-station hard seams. Each 64px world cell can
/// contribute at most one independently detected two-sided stroke pair; this
/// keeps the sample population spatially deterministic and prevents jagged
/// one-pixel owner transitions from inflating the criterion.
fn measure_large_boundaries(
    image: &Rgb32FImage,
    coverage: &GrayImage,
    ownership: &[u16],
    origin: (f64, f64),
    sources: &[QualitySource],
    accepted_station_relations: &[(usize, usize)],
    residual: &ResidualWarp,
    criterion: &mut Criterion,
    unmeasurable: &mut Vec<UnmeasurableRecord>,
) {
    const SPATIAL_DEDUP_PX: u32 = 64;
    let (width, height) = coverage.dimensions();
    let width_usize = width as usize;
    let source_by_owner = sources
        .iter()
        .map(|source| (source.owner, source))
        .collect::<BTreeMap<_, _>>();
    let accepted = accepted_station_relations
        .iter()
        .filter_map(|&(left, right)| (left != right).then_some((left.min(right), left.max(right))))
        .collect::<std::collections::BTreeSet<_>>();
    let mut cells = BTreeMap::<(usize, usize, u32, u32), BoundaryCellObservation>::new();
    let mut provenance_unavailable = false;
    let mut visit = |x: u32, y: u32, normal: (f64, f64), left_owner: u16, right_owner: u16| {
        if left_owner == 0 || right_owner == 0 || left_owner == right_owner {
            return;
        }
        let (Some(left_source), Some(right_source)) = (
            source_by_owner.get(&left_owner).copied(),
            source_by_owner.get(&right_owner).copied(),
        ) else {
            provenance_unavailable = true;
            return;
        };
        let station_pair = (
            left_source
                .geometry
                .station_id
                .min(right_source.geometry.station_id),
            left_source
                .geometry
                .station_id
                .max(right_source.geometry.station_id),
        );
        if station_pair.0 == station_pair.1 || !accepted.contains(&station_pair) {
            return;
        }
        let seam = (f64::from(x) + normal.0 * 0.5, f64::from(y) + normal.1 * 0.5);
        let world = (origin.0 + seam.0, origin.1 + seam.1);
        let seam_key = (
            station_pair.0,
            station_pair.1,
            x / SPATIAL_DEDUP_PX,
            y / SPATIAL_DEDUP_PX,
        );
        let (key, observation) = match boundary_pair_error(
            image,
            coverage,
            ownership,
            seam,
            normal,
            left_owner,
            right_owner,
        ) {
            Ok((left_edge, right_edge)) => {
                let left_world = (
                    origin.0 + f64::from(left_edge.pixel.0),
                    origin.1 + f64::from(left_edge.pixel.1),
                );
                let right_world = (
                    origin.0 + f64::from(right_edge.pixel.0),
                    origin.1 + f64::from(right_edge.pixel.1),
                );
                let stroke_key = (
                    station_pair.0,
                    station_pair.1,
                    (left_edge.pixel.0 / 2 + right_edge.pixel.0 / 2) / SPATIAL_DEDUP_PX,
                    (left_edge.pixel.1 / 2 + right_edge.pixel.1 / 2) / SPATIAL_DEDUP_PX,
                );
                if !source_contains_world(left_source, left_world, residual)
                    || !source_contains_world(right_source, right_world, residual)
                {
                    (
                        stroke_key,
                        BoundaryCellObservation::Unmeasurable {
                            world,
                            reason: degradation::BOUNDARY_PROVENANCE_UNAVAILABLE,
                        },
                    )
                } else {
                    (
                        stroke_key,
                        BoundaryCellObservation::Measured(BoundaryPairEvidence {
                            world,
                            error: (left_edge.offset - right_edge.offset).abs(),
                            strength: left_edge.strength.min(right_edge.strength),
                            owner_path: format!(
                                "station:{}/station:{};owner:{}/owner:{};samples:{:.1},{:.1}|{:.1},{:.1}",
                                station_pair.0,
                                station_pair.1,
                                left_owner,
                                right_owner,
                                left_world.0,
                                left_world.1,
                                right_world.0,
                                right_world.1,
                            ),
                        }),
                    )
                }
            }
            Err(reason) => (
                seam_key,
                BoundaryCellObservation::Unmeasurable { world, reason },
            ),
        };
        match (cells.get(&key), &observation) {
            (
                Some(BoundaryCellObservation::Measured(current)),
                BoundaryCellObservation::Measured(candidate),
            ) if candidate.strength > current.strength => {
                cells.insert(key, observation);
            }
            (Some(BoundaryCellObservation::Measured(_)), _) => {}
            (
                Some(BoundaryCellObservation::Unmeasurable {
                    reason: current, ..
                }),
                BoundaryCellObservation::Unmeasurable {
                    reason: candidate, ..
                },
            ) if boundary_rejection_priority(candidate) > boundary_rejection_priority(current) => {
                cells.insert(key, observation);
            }
            (_, BoundaryCellObservation::Measured(_)) | (None, _) => {
                cells.insert(key, observation);
            }
            _ => {}
        }
    };
    for y in 0..height {
        for x in 0..width.saturating_sub(1) {
            let index = (y * width + x) as usize;
            if coverage.as_raw()[index] != 0
                && coverage.as_raw()[index + 1] != 0
                && ownership[index] != ownership[index + 1]
            {
                visit(x, y, (1.0, 0.0), ownership[index], ownership[index + 1]);
            }
        }
    }
    for y in 0..height.saturating_sub(1) {
        for x in 0..width {
            let index = (y * width + x) as usize;
            if coverage.as_raw()[index] != 0
                && coverage.as_raw()[index + width_usize] != 0
                && ownership[index] != ownership[index + width_usize]
            {
                visit(
                    x,
                    y,
                    (0.0, 1.0),
                    ownership[index],
                    ownership[index + width_usize],
                );
            }
        }
    }

    let mut errors = cells
        .values()
        .filter_map(|observation| match observation {
            BoundaryCellObservation::Measured(evidence) => Some(evidence.error),
            BoundaryCellObservation::Unmeasurable { .. } => None,
        })
        .collect::<Vec<_>>();
    errors.sort_by(f64::total_cmp);
    let p95 = (!errors.is_empty()).then(|| {
        errors[(errors.len() * 95)
            .div_ceil(100)
            .saturating_sub(1)
            .min(errors.len() - 1)]
    });
    for observation in cells.into_values() {
        match observation {
            BoundaryCellObservation::Measured(evidence) => {
                let p95_failed = p95.is_some_and(|value| {
                    value > quality_gate::QUALITY_BOUNDARY_P95_MAX && evidence.error >= value
                });
                criterion.push(
                    evidence.world,
                    evidence.error,
                    &evidence.owner_path,
                    evidence.error <= quality_gate::QUALITY_BOUNDARY_MAX && !p95_failed,
                );
            }
            BoundaryCellObservation::Unmeasurable { world, reason } => {
                criterion.miss(reason);
                unmeasurable.push(UnmeasurableRecord {
                    criterion: criterion.name.to_string(),
                    world: WorldPoint {
                        x: world.0,
                        y: world.1,
                    },
                    reason: reason.to_string(),
                    category: classify_unmeasurable_reason(reason),
                });
            }
        }
    }
    if criterion.observed_count == 0 && criterion.unmeasurable == 0 {
        let reason = if provenance_unavailable {
            degradation::BOUNDARY_PROVENANCE_UNAVAILABLE
        } else {
            degradation::BOUNDARY_NO_PAIRABLE_EDGE
        };
        criterion.miss(reason);
        unmeasurable.push(UnmeasurableRecord {
            criterion: criterion.name.to_string(),
            world: WorldPoint {
                x: origin.0,
                y: origin.1,
            },
            reason: reason.to_string(),
            category: classify_unmeasurable_reason(reason),
        });
    }
}
