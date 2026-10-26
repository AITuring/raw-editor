//! Virtual-tile prior repair and render installation.

use super::*;

/// Plan rendering-prior repairs from prior-free proposals.
///
/// Stations joined by authoritative (prior-seeded) relations form clusters
/// whose priors agree. The largest cluster is the reference (ties: smallest
/// canonical row/column key). Each proposal joining a fixed cluster to an
/// unfixed one — strongest first, ties by canonical pair key — moves the whole
/// unfixed cluster so that the proposal's measured relation holds; a cluster is
/// repaired at most once and a station therefore re-rendered at most once.
/// Stations no proposal reaches keep their prior (and stay disconnected).
pub(crate) fn plan_virtual_tile_prior_repairs_with_centres(
    station_count: usize,
    authoritative_pairs: &[(usize, usize)],
    proposals: &[VirtualTilePriorProposal],
    tile_to_world: &[Matrix3<f64>],
    station_world_centres: &[Point2<f64>],
    station_topology: &topology::StationTopology,
) -> Vec<VirtualTilePriorRepair> {
    if station_count < 2
        || tile_to_world.len() != station_count
        || station_world_centres.len() != station_count
    {
        return Vec::new();
    }
    let mut clusters = Dsu::new(station_count);
    for &(left, right) in authoritative_pairs {
        if left < station_count && right < station_count && left != right {
            clusters.union(left, right);
        }
    }
    let mut members = BTreeMap::<usize, Vec<usize>>::new();
    for station in 0..station_count {
        members
            .entry(clusters.find(station))
            .or_default()
            .push(station);
    }
    if members.len() < 2 {
        return Vec::new();
    }
    let cluster_key = |stations: &[usize]| {
        stations
            .iter()
            .map(|&station| station_tree_position_key(station_topology, station))
            .min()
    };
    let Some(reference) = members
        .iter()
        .max_by(|(_, left), (_, right)| {
            left.len()
                .cmp(&right.len())
                .then_with(|| cluster_key(right).cmp(&cluster_key(left)))
        })
        .map(|(&root, _)| root)
    else {
        return Vec::new();
    };
    let mut corrections = BTreeMap::from([(reference, Matrix3::identity())]);
    let mut remaining = proposals
        .iter()
        .copied()
        .filter(|proposal| {
            proposal.left < station_count
                && proposal.right < station_count
                && proposal.left != proposal.right
        })
        .collect::<Vec<_>>();
    let mut repairs = Vec::new();
    loop {
        let next = remaining
            .iter()
            .enumerate()
            .filter(|(_, proposal)| {
                corrections.contains_key(&clusters.find(proposal.left))
                    != corrections.contains_key(&clusters.find(proposal.right))
            })
            .max_by(|(_, left), (_, right)| {
                left.inliers.cmp(&right.inliers).then_with(|| {
                    station_tree_edge_key(station_topology, right.left, right.right).cmp(
                        &station_tree_edge_key(station_topology, left.left, left.right),
                    )
                })
            })
            .map(|(index, _)| index);
        let Some(index) = next else {
            break;
        };
        let proposal = remaining.remove(index);
        let left_root = clusters.find(proposal.left);
        let right_root = clusters.find(proposal.right);
        // tile pixel q of the unfixed station lands where the fixed station's
        // (already corrected) placement puts the measured corresponding pixel.
        let (fixed, unfixed, unfixed_root, fixed_to_unfixed_tile) =
            if corrections.contains_key(&left_root) {
                (
                    proposal.left,
                    proposal.right,
                    right_root,
                    proposal.left_to_right.try_inverse(),
                )
            } else {
                (
                    proposal.right,
                    proposal.left,
                    left_root,
                    Some(proposal.left_to_right),
                )
            };
        let Some(unfixed_to_fixed) = fixed_to_unfixed_tile else {
            continue;
        };
        let fixed_correction = corrections[&clusters.find(fixed)];
        let Some(unfixed_world_to_tile) = tile_to_world[unfixed].try_inverse() else {
            continue;
        };
        let Some(correction) = normalized_homography(
            &(fixed_correction * tile_to_world[fixed] * unfixed_to_fixed * unfixed_world_to_tile),
        ) else {
            continue;
        };
        if correction.try_inverse().is_none()
            || members[&unfixed_root].iter().any(|&station| {
                !virtual_tile_prior_repair_is_bounded_at_center(
                    &correction,
                    station_world_centres[station],
                )
            })
        {
            // Reject the proposal atomically before this cluster is marked as
            // repaired, so a weaker bounded proposal remains eligible.
            continue;
        }
        corrections.insert(unfixed_root, correction);
        for &station in &members[&unfixed_root] {
            repairs.push(VirtualTilePriorRepair {
                station,
                cluster_root: unfixed_root,
                fixed_station: fixed,
                relation: (proposal.left, proposal.right),
                relation_inliers: proposal.inliers,
                world_correction: correction,
            });
        }
    }
    repairs.sort_by_key(|repair| repair.station);
    repairs
}

#[cfg(test)]
pub(crate) fn plan_virtual_tile_prior_repairs(
    station_count: usize,
    authoritative_pairs: &[(usize, usize)],
    proposals: &[VirtualTilePriorProposal],
    tile_to_world: &[Matrix3<f64>],
    station_topology: &topology::StationTopology,
) -> Vec<VirtualTilePriorRepair> {
    let station_world_centres = tile_to_world
        .iter()
        .filter_map(|pose| transformed_point(pose, Point2::new(500.0, 400.0)))
        .collect::<Vec<_>>();
    plan_virtual_tile_prior_repairs_with_centres(
        station_count,
        authoritative_pairs,
        proposals,
        tile_to_world,
        &station_world_centres,
        station_topology,
    )
}

/// Render one Capture_Station for station matching only and return its covered
/// luminance plane and Coverage_Mask; the fused colour pixels are dropped.
#[allow(clippy::too_many_arguments)]
pub(crate) fn render_virtual_tile_match_plane<R: Runtime, F>(
    group: &[&ImageInfo],
    homographies: &HashMap<usize, Matrix3<f64>>,
    projection: Projection,
    focus_warp: Option<&FocusLayerWarp>,
    station_index: usize,
    expected: stitching::FocusVirtualTileGeometry,
    app_handle: AppHandle<R>,
    progress_event: &str,
    load_image: &mut F,
) -> Result<(GrayImage, stitching::CoverageMask), String>
where
    F: FnMut(&ImageInfo) -> Result<Rgb32FImage, String>,
{
    let rendered = stitching::focus_stack_stitcher_unfilled(
        group,
        homographies,
        projection,
        focus_warp,
        None,
        station_index,
        false,
        app_handle,
        progress_event,
        load_image,
    )?;
    let rendered_dimensions = rendered.image.dimensions();
    if rendered_dimensions != (expected.width, expected.height) {
        return Err(format!(
            "Virtual tile station {station_index} geometry/render mismatch: expected {}x{} at world origin ({:.3},{:.3}), rendered {}x{}",
            expected.width,
            expected.height,
            expected.tile_to_world[(0, 2)],
            expected.tile_to_world[(1, 2)],
            rendered_dimensions.0,
            rendered_dimensions.1,
        ));
    }
    let coverage = match rendered.masks.as_ref() {
        Some(masks) => masks.coverage.clone(),
        None => stitching::CoverageMask::from_bytes(
            expected.width,
            expected.height,
            vec![0; expected.width as usize * expected.height as usize],
        )?,
    };
    let gray = virtual_tile_covered_luminance(&rendered.image, &coverage);
    Ok((gray, coverage))
}

/// Install one station's probe planes. The tile keeps the structural
/// relation-quality plane as its analysis image (`scale_factor` = that plane's
/// factor, the source `ImageInfo` convention) and independent covered features.
/// Small preview probe grids must not downsample feature descriptors twice.
/// The feature factor is explicit; nothing reads keypoints in quality-plane
/// coordinates or promotes a feature fit to solver evidence.
pub(crate) fn install_virtual_tile_match_planes(
    tile: &mut ImageInfo,
    pyramid: &mut VirtualTileMatchPyramid,
    station: usize,
    gray: &GrayImage,
    coverage: &stitching::CoverageMask,
    brief_pairs: &[(Point2<i32>, Point2<i32>)],
) -> Result<(), String> {
    let planes = pyramid
        .set_station(station, tile, gray, coverage)
        .ok_or_else(|| format!("Virtual tile station {station} has no usable matching plane"))?;
    let finest = planes
        .last()
        .expect("set_station returns at least one plane");
    let feature_factor = finest
        .factor
        .min((gray.width().max(gray.height()) / 1200).max(1));
    let feature_plane = VirtualTileMatchPlane::from_native(gray, coverage, feature_factor)
        .ok_or_else(|| format!("Virtual tile station {station} has no feature proposal plane"))?;
    tile.features = virtual_tile_plane_features(&feature_plane, brief_pairs);
    pyramid.feature_factors[station] = feature_factor;
    let (quality, factor) = virtual_tile_relation_quality_plane(&planes, gray, coverage);
    #[cfg(test)]
    if let Some(directory) = std::env::var_os("RAW_EDITOR_VIRTUAL_TILE_DIAGNOSTICS_DIR") {
        let directory = PathBuf::from(directory);
        std::fs::create_dir_all(&directory).map_err(|error| error.to_string())?;
        gray.save(directory.join(format!("station-{station:03}.gray.png")))
            .map_err(|error| error.to_string())?;
        quality
            .save(directory.join(format!("station-{station:03}.quality.png")))
            .map_err(|error| error.to_string())?;
        let (w, h) = coverage.dimensions();
        GrayImage::from_raw(w, h, coverage.covered().to_vec())
            .ok_or("invalid diagnostic coverage")?
            .save(directory.join(format!("station-{station:03}.coverage.png")))
            .map_err(|error| error.to_string())?;
    }
    let quality_coverage = planes
        .iter()
        .find(|plane| plane.factor == factor)
        .map(|plane| plane.coverage.clone())
        .or_else(|| {
            VirtualTileMatchPlane::from_native(gray, coverage, factor).map(|plane| plane.coverage)
        })
        .ok_or_else(|| format!("Virtual tile station {station} has no relation-quality plane"))?;
    pyramid.quality_coverages[station] = quality_coverage;
    tile.scale_factor = f64::from(factor);
    tile.alignment_image = quality;
    Ok(())
}

/// Move one station's rendering prior: every member source pose (and its
/// focus-band poses) is composed with `world_correction`. Returns false, and
/// changes nothing, when a corrected pose would not be a valid planar pose.
pub(crate) fn apply_virtual_tile_prior_repair(
    group: &[&ImageInfo],
    world_correction: &Matrix3<f64>,
    homographies: &mut HashMap<usize, Matrix3<f64>>,
    focus_warp: Option<&mut FocusLayerWarp>,
) -> bool {
    let mut corrected = Vec::with_capacity(group.len());
    for image in group {
        let Some(pose) = homographies.get(&image.id) else {
            return false;
        };
        let Some(pose) = normalized_homography(&(world_correction * pose)) else {
            return false;
        };
        if pose.try_inverse().is_none()
            || !homography_preserves_focus_orientation(&pose, image.dimensions())
        {
            return false;
        }
        corrected.push((image.id, pose));
    }
    for (id, pose) in corrected {
        homographies.insert(id, pose);
    }
    if let Some(warp) = focus_warp {
        for band in &mut warp.bands {
            for image in group {
                if let Some(pose) = band.homographies.get_mut(&image.id)
                    && let Some(corrected) = normalized_homography(&(world_correction * *pose))
                {
                    *pose = corrected;
                }
            }
        }
    }
    true
}

pub(crate) fn replace_virtual_tile_candidate_record(
    records: &mut Vec<stack_report::StationRelationCandidateRecord>,
    record: stack_report::StationRelationCandidateRecord,
) {
    let key = (record.left.min(record.right), record.left.max(record.right));
    match records.iter_mut().find(|existing| {
        (
            existing.left.min(existing.right),
            existing.left.max(existing.right),
        ) == key
    }) {
        Some(existing) => *existing = record,
        None => records.push(record),
    }
}

/// Fit station relations directly from focus-fused Virtual_Tiles. The topology
/// controls deterministic attempt order and budget. Initial poses select the
/// jointly covered grid and predict bounded target search centres only; every
/// returned point and fitted model is measured directly from fused tile pixels,
/// coarse to fine on area-averaged planes of the supplied native planes.
/// Production drives the same measurement through
/// [`virtual_tile_measure_prior_relations`] with its per-station render loop.
#[cfg(test)]
pub(crate) fn virtual_tile_station_match_outcome(
    tiles: &[ImageInfo],
    coverages: &[stitching::CoverageMask],
    initial_homographies: &HashMap<usize, Matrix3<f64>>,
    station_topology: &topology::StationTopology,
) -> VirtualTileStationMatchOutcome {
    let shortest_long_side = tiles
        .iter()
        .map(|tile| {
            tile.alignment_image
                .width()
                .max(tile.alignment_image.height())
        })
        .min()
        .unwrap_or(0);
    let mut pyramid = VirtualTileMatchPyramid::new(
        virtual_tile_pyramid_factors(shortest_long_side),
        tiles.len(),
    );
    for (station, tile) in tiles.iter().enumerate() {
        if let Some(coverage) = coverages.get(station) {
            pyramid.set_station(station, tile, &tile.alignment_image, coverage);
        }
    }
    let (relations, diagnostics) = virtual_tile_measure_prior_relations(
        tiles,
        &pyramid,
        initial_homographies,
        station_topology,
        None,
        false,
    );
    VirtualTileStationMatchOutcome {
        relations,
        diagnostics,
    }
}

#[cfg(test)]
pub(crate) fn virtual_tile_station_matches(
    tiles: &[ImageInfo],
    coverages: &[stitching::CoverageMask],
    initial_homographies: &HashMap<usize, Matrix3<f64>>,
    station_topology: &topology::StationTopology,
) -> HashMap<(usize, usize), MatchInfo> {
    let outcome = virtual_tile_station_match_outcome(
        tiles,
        coverages,
        initial_homographies,
        station_topology,
    );
    select_virtual_tile_relation_hypotheses(
        tiles,
        &outcome.relations,
        VirtualTileSolverCoverages {
            quality: coverages,
            overlap: coverages,
            overlap_factor: 1,
        },
    )
}
