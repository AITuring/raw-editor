//! Property 90: the reachable determinism-critical subset of a run is byte for
//! byte reproducible.
//!
//! Run seed, pairwise RANSAC, station poses, Virtual_Tile placement, compositor
//! order and the canvas reduction, across repeats, thread counts and import
//! permutations.

use super::super::determinism::DETERMINISTIC_SUM_BLOCK_LEN;
use super::super::determinism::derive_run_seed_from_paths;
use super::super::report::ConnectivityReport;
use super::super::test_support::SyntheticScan;
use super::super::test_support::arb_artwork_scan_grid;
use crate::image_stack::STACK_PIPELINE_VERSION;
use crate::panorama_stitching::determinism_test_access;
use crate::panorama_utils::processing;
use crate::panorama_utils::registration;
use crate::panorama_utils::shared::BRIEF_DESCRIPTOR_SIZE;
use crate::panorama_utils::shared::Feature;
use crate::panorama_utils::shared::ImageInfo;
use crate::panorama_utils::shared::KeyPoint;
use crate::panorama_utils::shared::MatchInfo;
use crate::panorama_utils::stitching;
use nalgebra::Matrix3;
use nalgebra::Point2;
use proptest::prelude::*;
use std::collections::BTreeSet;
use std::collections::HashMap;
use std::sync::OnceLock;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

use super::map_point;

// ---------------------------------------------------------------------------
// Property 90 harness: the reachable determinism critical subset of a run
// ---------------------------------------------------------------------------
//
// `stitch_images_with_options` needs an `AppHandle`, decodable files on disk and
// a Tauri event channel, so it cannot run inside `cargo test --lib`.  The
// harness below therefore drives the largest subset of the real pipeline that a
// unit test can reach, in the production order:
//
//   1. `determinism::derive_run_seed_from_paths` — the content derived RANSAC
//      seed of the run (设计「时间与随机源」).
//   2. `canonical_match_direction` mirrored, then
//      `processing::find_homography_ransac_points_stable_with_min_inliers` —
//      the real RANSAC pairwise fit for every overlapping pair.
//   3. `solve_focus_capture_group_poses` — the station pose solve, reached
//      through `determinism_test_access`.  This is the function task 3.1 moved
//      off `HashMap` iteration, task 3.2 gave a fixed reduction order and
//      task 3.4 gave a world-coordinate sort tie-break.
//   4. `focus_capture_groups` + `stitching::focus_stack_virtual_tile_geometry` —
//      station membership and the pixel aligned Virtual_Tile placement.
//   5. `focus_source_order_group_compositor_order` — the seam/compositor order.
//   6. `determinism::deterministic_sum` over the canvas samples of every
//      placed station — the blocked floating point reduction.
//
// What stays out of reach here and is covered by the stage 8 Acceptance_Harness
// instead: RAW decoding, focus fusion pixels, the tone harmonizer, the seam
// graph cut, Virtual_Tile cache hits and the encoded output file itself.

/// Inlier threshold, in tile pixels, of the synthetic pairwise fits.
pub(super) const SYNTHETIC_RANSAC_INLIER_THRESHOLD_PX: f64 = 2.0;

/// Minimum inliers demanded of a synthetic pairwise fit.
pub(super) const SYNTHETIC_RANSAC_MIN_INLIERS: usize = 8;

/// Canvas samples reduced per placed station.
///
/// Above [`DETERMINISTIC_SUM_BLOCK_LEN`] on purpose, so the reduction takes the
/// blocked `rayon` path rather than the serial short-circuit — that is the path
/// whose independence from the thread count has to be observable.
pub(super) const CANVAS_SAMPLES_PER_STATION: usize = 96 * 64;

/// Sample grid columns inside one station tile.
pub(super) const CANVAS_SAMPLE_COLUMNS: usize = 96;

/// Geometry result of the production station solver.
///
/// A disconnected graph has no renderable pose/output result.  Its canonical
/// component member sets are retained so rename/import permutations can be
/// compared semantically instead of accidentally treating the old locked-pose
/// fallback as a successful solve.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum SyntheticGeometryOutcome {
    Connected,
    RejectedDisconnected {
        components: BTreeSet<BTreeSet<(usize, usize)>>,
        member_counts: Vec<usize>,
    },
}

pub(super) fn synthetic_geometry_outcome(
    images: &[ImageInfo],
    identities: &[(usize, usize)],
    solve: &determinism_test_access::StationPoseSolveView,
) -> SyntheticGeometryOutcome {
    let ConnectivityReport {
        components,
        component_members,
        component_member_counts,
    } = &solve.connectivity;
    assert_eq!(*components, component_members.len());
    assert_eq!(component_members.len(), component_member_counts.len());

    if *components <= 1 {
        assert!(
            !solve.poses.is_empty(),
            "connected synthetic station geometry must retain solved poses"
        );
        return SyntheticGeometryOutcome::Connected;
    }

    assert!(
        solve.poses.is_empty(),
        "requirements 6.7/12.4 reject disconnected geometry instead of returning locked poses"
    );
    assert!(!solve.translation_geometry_verified);
    let mut canonical_counts = component_member_counts.clone();
    canonical_counts.sort_unstable();
    let canonical_components = component_members
        .iter()
        .map(|members| {
            members
                .iter()
                .map(|path| {
                    let image_index = images
                        .iter()
                        .position(|image| image.filename == *path)
                        .expect("every disconnected report path names an input source");
                    identities[image_index]
                })
                .collect::<BTreeSet<_>>()
        })
        .collect::<BTreeSet<_>>();
    let mut set_counts = canonical_components
        .iter()
        .map(BTreeSet::len)
        .collect::<Vec<_>>();
    set_counts.sort_unstable();
    assert_eq!(canonical_counts, set_counts);

    SyntheticGeometryOutcome::RejectedDisconnected {
        components: canonical_components,
        member_counts: canonical_counts,
    }
}

/// Result of one synthetic run.
pub(super) struct SyntheticRun {
    /// Canonical bytes of every successful output decision plus every pixel
    /// aligned quantity it produced. Disconnected runs have no output bytes.
    pub(super) placement: Option<Vec<u8>>,
    /// Solved world homography per source, keyed by file name so that the
    /// encoding cannot inherit the import order. Rejected runs keep this empty.
    pub(super) poses: Vec<(String, Matrix3<f64>)>,
    pub(super) geometry: SyntheticGeometryOutcome,
    /// Coverage witnesses: without them a degenerate run (fewer than two
    /// stations, no accepted relation) would satisfy every assertion below
    /// without ever entering the code under test.
    pub(super) stations: usize,
    pub(super) station_relations: usize,
    pub(super) bracket_relations: usize,
}

pub(super) fn push_f64(bytes: &mut Vec<u8>, value: f64) {
    bytes.extend_from_slice(&value.to_bits().to_be_bytes());
}

pub(super) fn push_matrix(bytes: &mut Vec<u8>, matrix: &Matrix3<f64>) {
    for value in matrix.iter() {
        push_f64(bytes, *value);
    }
}

pub(super) fn push_str(bytes: &mut Vec<u8>, text: &str) {
    bytes.extend_from_slice(&(text.len() as u64).to_be_bytes());
    bytes.extend_from_slice(text.as_bytes());
}

/// Add stable feature signatures to the determinism fixtures.
///
/// Real decoded images reach `canonical_match_direction` with detected feature
/// content.  The base synthetic generator intentionally leaves features empty
/// for grouping tests, which activates a filename fallback and makes a rename
/// change the asymmetric RANSAC direction. These signatures model the real
/// content-derived path without changing any pixels, poses, or correspondences.
pub(super) fn determinism_images(scan: &SyntheticScan) -> Vec<ImageInfo> {
    let mut images = scan.images();
    for (image, &(station, layer)) in images.iter_mut().zip(scan.capture_identities().iter()) {
        let mut descriptor = [0u8; BRIEF_DESCRIPTOR_SIZE / 8];
        descriptor[..8].copy_from_slice(&(station as u64).to_be_bytes());
        descriptor[8..16].copy_from_slice(&(layer as u64).to_be_bytes());
        image.features.push(Feature {
            keypoint: KeyPoint {
                x: 24 + station as u32,
                y: 24 + layer as u32,
            },
            descriptor,
            support_scale: 1.0,
        });
    }
    images
}

/// Fit every overlapping pair with the real RANSAC entry point.
///
/// The measurement direction is exactly the production
/// `canonical_match_direction`: content-derived when both images carry feature
/// signatures, with its documented filename fallback for deliberately
/// featureless fixtures. The result is inverted back into the `(min, max)` key
/// orientation exactly like `stitch_images_with_options` does.
pub(super) fn fit_synthetic_matches(
    scan: &SyntheticScan,
    images: &[ImageInfo],
) -> (
    HashMap<(usize, usize), MatchInfo>,
    Vec<(String, String, Matrix3<f64>)>,
) {
    let mut matches = HashMap::new();
    let mut measured = Vec::new();
    for ((left, right), points) in scan.correspondence_pairs() {
        let (source, target, invert_for_storage) =
            determinism_test_access::canonical_pair_direction(images, left, right);
        let measurement = if invert_for_storage {
            points
                .iter()
                .map(|&(source, target)| (target, source))
                .collect::<Vec<_>>()
        } else {
            points.clone()
        };
        let Some((homography, inlier_indices)) =
            processing::find_homography_ransac_points_stable_with_min_inliers(
                &measurement,
                SYNTHETIC_RANSAC_INLIER_THRESHOLD_PX,
                SYNTHETIC_RANSAC_MIN_INLIERS,
            )
        else {
            continue;
        };
        let inliers = inlier_indices
            .iter()
            .filter_map(|&index| measurement.get(index).copied())
            .collect::<Vec<_>>();
        if inliers.len() < SYNTHETIC_RANSAC_MIN_INLIERS {
            continue;
        }
        let source_name = images[source].filename.clone();
        let target_name = images[target].filename.clone();
        measured.push((source_name, target_name, homography));
        let (stored_homography, stored_points) = if invert_for_storage {
            let Some(inverse) = homography.try_inverse() else {
                continue;
            };
            (
                inverse,
                inliers
                    .iter()
                    .map(|&(source, target)| (target, source))
                    .collect::<Vec<_>>(),
            )
        } else {
            (homography, inliers)
        };
        matches.insert(
            (left, right),
            MatchInfo {
                // Mirror the production storage step: the fit is measured in the
                // canonical direction and kept there, while `homography` is
                // rotated into the `(min index, max index)` key orientation.
                canonical_homography: Some(homography),
                native_verified_points: Vec::new(),
                homography: stored_homography,
                inliers: stored_points.len(),
                sequence_bridge: false,
                coarse_bridge: false,
                points: stored_points,
                candidate_points: Vec::new(),
                top_candidate_points: Vec::new(),
                dense_focus_points: Vec::new(),
                foreground_feature_points: Vec::new(),
            },
        );
    }
    (matches, measured)
}

/// Canvas samples of one placed station, weighted across 40 decades.
///
/// The weight spread is what makes the *reduction order* observable at all: a
/// sum of same-magnitude values can round to the same bits whatever shape the
/// reduction tree has, so an unweighted sum would pass even for
/// `par_iter().sum()`.  This is the same device the `determinism` unit tests
/// use to prove their own assertions can fail.
pub(super) fn station_canvas_samples(
    plane: &image::GrayImage,
    geometry: &stitching::FocusVirtualTileGeometry,
    samples: &mut Vec<f64>,
) {
    let rows = CANVAS_SAMPLES_PER_STATION / CANVAS_SAMPLE_COLUMNS;
    for index in 0..CANVAS_SAMPLES_PER_STATION {
        let column = index % CANVAS_SAMPLE_COLUMNS;
        let row = index / CANVAS_SAMPLE_COLUMNS;
        let tile = Point2::new(
            (column as f64 + 0.5) * f64::from(geometry.width) / CANVAS_SAMPLE_COLUMNS as f64,
            (row as f64 + 0.5) * f64::from(geometry.height) / rows as f64,
        );
        let value = map_point(&geometry.tile_to_world, tile)
            .and_then(|world| registration::sample_gray(plane, world.x, world.y))
            .unwrap_or(0.0);
        let decade = (index % 40) as i32 - 20;
        let sign = if index % 3 == 0 { -1.0 } else { 1.0 };
        samples.push(sign * (1.0 + value) * 10f64.powi(decade));
    }
}

pub(super) fn run_synthetic_stack(scan: &SyntheticScan) -> SyntheticRun {
    let images = determinism_images(scan);
    let locked = scan.locked_homographies();

    // 1. The RANSAC seed of this run, derived from the selection content.
    let run_seed = derive_run_seed_from_paths(STACK_PIPELINE_VERSION, &scan.source_paths());

    // 2. Real pairwise RANSAC.  `matches` is a real `HashMap` filled in import
    //    order, so its iteration order is exactly the hash order the pipeline
    //    is forbidden to follow.
    let (matches, mut measured) = fit_synthetic_matches(scan, &images);
    let mut station_relations = 0usize;
    let mut bracket_relations = 0usize;
    for pair in {
        let mut keys = matches.keys().copied().collect::<Vec<_>>();
        keys.sort_unstable();
        keys
    } {
        if scan.sources()[pair.0].station == scan.sources()[pair.1].station {
            bracket_relations += 1;
        } else {
            station_relations += 1;
        }
    }

    // 3. The station pose solve. A disconnected graph is an explicit rejected
    // outcome and therefore contributes no renderable poses or output bytes.
    let solve = determinism_test_access::solve_station_poses_with_report(
        &images,
        &matches,
        &locked,
        scan.reference_index(),
    );
    let identities = scan.capture_identities();
    let geometry = synthetic_geometry_outcome(&images, &identities, &solve);
    let grouping_poses = match &geometry {
        SyntheticGeometryOutcome::Connected => &solve.poses,
        SyntheticGeometryOutcome::RejectedDisconnected { .. } => &locked,
    };

    // 4. Station membership is established before station-graph rejection. The
    // locked image geometry remains valid for observing that grouping on a
    // rejected run, but must never be presented as a solved/output geometry.
    let stations = determinism_test_access::capture_stations(&images, &matches, grouping_poses)
        .unwrap_or_default();
    let geometries = if geometry == SyntheticGeometryOutcome::Connected {
        stations
            .iter()
            .map(|station| {
                let members = station
                    .members
                    .iter()
                    .filter_map(|&index| images.get(index))
                    .collect::<Vec<_>>();
                stitching::focus_stack_virtual_tile_geometry(
                    &members,
                    &solve.poses,
                    stitching::Projection::Planar,
                    None,
                )
            })
            .collect::<Option<Vec<_>>>()
            .unwrap_or_default()
    } else {
        Vec::new()
    };

    // 5. Only connected station geometry reaches the compositor.
    let compositor_order =
        if geometry == SyntheticGeometryOutcome::Connected && geometries.len() == stations.len() {
            determinism_test_access::station_compositor_order(&images, &stations, &solve.poses)
        } else {
            Vec::new()
        };

    let placement = if geometry == SyntheticGeometryOutcome::Connected {
        let mut samples = Vec::with_capacity(compositor_order.len() * CANVAS_SAMPLES_PER_STATION);
        for &station in &compositor_order {
            station_canvas_samples(scan.plane().image(), &geometries[station], &mut samples);
        }
        let canvas_reduction =
            crate::panorama_utils::stack_pipeline::determinism::deterministic_sum(&samples);

        let mut placement = Vec::new();
        placement.extend_from_slice(&run_seed.to_be_bytes());
        placement.push(u8::from(solve.translation_geometry_verified));
        placement.extend_from_slice(&(stations.len() as u64).to_be_bytes());
        for &station in &compositor_order {
            let geometry = &geometries[station];
            placement.extend_from_slice(&geometry.width.to_be_bytes());
            placement.extend_from_slice(&geometry.height.to_be_bytes());
            push_matrix(&mut placement, &geometry.tile_to_world);
            let mut members = stations[station]
                .members
                .iter()
                .filter_map(|&index| images.get(index))
                .map(|image| image.filename.clone())
                .collect::<Vec<_>>();
            members.sort();
            placement.extend_from_slice(&(members.len() as u64).to_be_bytes());
            for member in &members {
                push_str(&mut placement, member);
            }
        }
        push_f64(&mut placement, canvas_reduction);
        measured.sort_by(|left, right| (&left.0, &left.1).cmp(&(&right.0, &right.1)));
        placement.extend_from_slice(&(measured.len() as u64).to_be_bytes());
        for (source_name, target_name, homography) in &measured {
            push_str(&mut placement, source_name);
            push_str(&mut placement, target_name);
            push_matrix(&mut placement, homography);
        }
        Some(placement)
    } else {
        None
    };

    let mut poses = solve
        .poses
        .iter()
        .filter_map(|(&id, pose)| {
            images
                .iter()
                .find(|image| image.id == id)
                .map(|image| (image.filename.clone(), *pose))
        })
        .collect::<Vec<_>>();
    poses.sort_by(|left, right| left.0.cmp(&right.0));

    SyntheticRun {
        placement,
        poses,
        geometry,
        stations: stations.len(),
        station_relations,
        bracket_relations,
    }
}

/// Worker counts the thread dimension of Property 90 covers.
pub(super) const THREAD_DIMENSION: [usize; 4] = [1, 2, 8, 16];

/// Number of accepted Property 90 cases so far, which drives the rotation
/// through [`THREAD_DIMENSION`].
///
/// Running all four worker counts on every case costs four full harness passes
/// per case and buys nothing: the assertion is that the result is independent of
/// the worker count, and a counterexample would have to appear for *some* input
/// at *some* worker count. Rotating one worker count per case keeps the whole
/// dimension covered across the run (asserted below) at a quarter of the cost,
/// and `proptest` executes the cases of one test sequentially, so the rotation
/// is a deterministic function of the accepted case index.
pub(super) static THREAD_DIMENSION_CURSOR: AtomicUsize = AtomicUsize::new(0);

/// Bitmask of the [`THREAD_DIMENSION`] entries the accepted cases have run.
pub(super) static THREAD_DIMENSION_COVERED: AtomicUsize = AtomicUsize::new(0);

/// Fixed size `rayon` pools, built once and reused across cases.
///
/// Building a pool spawns and parks its workers, which is pure overhead repeated
/// 100 times per worker count otherwise. A pool is stateless between
/// `install` calls, so reuse cannot carry anything from one case to the next.
pub(super) fn shared_pool(threads: usize) -> &'static rayon::ThreadPool {
    static POOLS: OnceLock<HashMap<usize, rayon::ThreadPool>> = OnceLock::new();
    POOLS
        .get_or_init(|| {
            THREAD_DIMENSION
                .iter()
                .map(|&threads| {
                    let pool = rayon::ThreadPoolBuilder::new()
                        .num_threads(threads)
                        .build()
                        .expect("building a fixed size rayon pool must succeed");
                    (threads, pool)
                })
                .collect()
        })
        .get(&threads)
        .expect("every worker count of the thread dimension has a pool")
}

/// Run the harness inside a `rayon` pool of exactly `threads` workers.
///
/// A fixed size pool is used rather than `RAYON_NUM_THREADS` so the assertion
/// does not depend on the environment the suite happens to run in.
pub(super) fn run_synthetic_stack_on_threads(scan: &SyntheticScan, threads: usize) -> SyntheticRun {
    shared_pool(threads).install(|| run_synthetic_stack(scan))
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 100, ..ProptestConfig::default() })]

    // Feature: layered-camera-group-focus-stitching, Property 90: 对于任意相同
    // Source_RAW 集合与相同参数，在同一构建版本与同一平台上重复运行输出逐字节一致的最终
    // 结果文件，且该一致性不受并行线程数量、Virtual_Tile 缓存命中与否与文件导入顺序影响。
    //
    // **Validates: Requirements 14.6**
    #[test]
    fn stack_output_is_reproducible_byte_for_byte(scan in arb_artwork_scan_grid()) {
        let _run_scope = crate::panorama_utils::stack_pipeline::degradation::begin_run_scope();
        prop_assume!(scan.station_count() >= 2);

        let reference = run_synthetic_stack(&scan);

        // Coverage: the four determinism sources of the design are only
        // exercised when the station level solve really had stations and
        // relations to work with.
        prop_assert!(
            reference.stations >= 2,
            "the harness must reach the station level solve, got {} station(s)",
            reference.stations
        );
        prop_assert!(
            reference.station_relations >= 1,
            "the harness must accept at least one inter-station relation"
        );
        prop_assert!(
            reference.bracket_relations >= 1,
            "the harness must accept at least one focus bracket relation"
        );
        match &reference.geometry {
            SyntheticGeometryOutcome::Connected => {
                prop_assert!(reference.placement.is_some());
                prop_assert!(
                    CANVAS_SAMPLES_PER_STATION > DETERMINISTIC_SUM_BLOCK_LEN,
                    "connected output must take the blocked canvas reduction path"
                );
            }
            SyntheticGeometryOutcome::RejectedDisconnected { components, .. } => {
                prop_assert!(components.len() >= 2);
                prop_assert!(reference.placement.is_none());
                prop_assert!(reference.poses.is_empty());
            }
        }

        // 1. Repeating the same input reproduces the same bytes.  Two runs are
        //    separated in wall clock time, so a reintroduced time dependency in
        //    the RANSAC seed shows up right here.
        let repeated = run_synthetic_stack(&scan);
        assert_runs_identical(&reference, &repeated, "a repeated run")?;

        // 2. The thread count is invisible.  One worker count per accepted
        //    case, rotating through `THREAD_DIMENSION`; with `cases: 100` and a
        //    four entry dimension every entry runs 25 times, and the assertion
        //    below turns "the rotation really covered all four" into a test
        //    failure rather than a comment.
        let case = THREAD_DIMENSION_CURSOR.fetch_add(1, Ordering::Relaxed);
        let slot = case % THREAD_DIMENSION.len();
        let threads = THREAD_DIMENSION[slot];
        let covered = THREAD_DIMENSION_COVERED.fetch_or(1 << slot, Ordering::Relaxed) | (1 << slot);
        let threaded = run_synthetic_stack_on_threads(&scan, threads);
        assert_runs_identical(&reference, &threaded, &format!("{threads} rayon thread(s)"))?;
        prop_assert!(
            case + 1 < THREAD_DIMENSION.len()
                || covered == (1usize << THREAD_DIMENSION.len()) - 1,
            "case {} must have covered every worker count of {:?} by now, covered mask {:#b}",
            case,
            THREAD_DIMENSION,
            covered
        );

        // 3. Permuting the import order — same files, same poses, same
        //    exposures, different arrival order — does not change the result.
        let permuted_scan = scan.with_permuted_import_order();
        let permuted = run_synthetic_stack(&permuted_scan);
        assert_runs_identical(&reference, &permuted, "a permuted import order")?;
    }
}

/// Byte-for-byte comparison of two runs, with a readable first difference.
pub(super) fn assert_runs_identical(
    left: &SyntheticRun,
    right: &SyntheticRun,
    what: &str,
) -> Result<(), TestCaseError> {
    prop_assert_eq!(
        left.stations,
        right.stations,
        "{} changed the station count",
        what
    );
    prop_assert_eq!(
        left.station_relations,
        right.station_relations,
        "{} changed the accepted station relation count",
        what
    );
    prop_assert_eq!(
        left.bracket_relations,
        right.bracket_relations,
        "{} changed the accepted bracket relation count",
        what
    );
    prop_assert_eq!(
        &left.geometry,
        &right.geometry,
        "{} changed the accepted/rejected station-geometry outcome or disconnected component report",
        what
    );
    match (&left.geometry, &left.placement, &right.placement) {
        (SyntheticGeometryOutcome::Connected, Some(left_bytes), Some(right_bytes)) => {
            if left_bytes != right_bytes {
                let first_difference = left_bytes
                    .iter()
                    .zip(right_bytes)
                    .position(|(left_byte, right_byte)| left_byte != right_byte)
                    .unwrap_or_else(|| left_bytes.len().min(right_bytes.len()));
                prop_assert!(
                    false,
                    "{} changed the placement bytes at offset {} ({} vs {} bytes)",
                    what,
                    first_difference,
                    left_bytes.len(),
                    right_bytes.len()
                );
            }
        }
        (SyntheticGeometryOutcome::RejectedDisconnected { .. }, None, None) => {
            prop_assert!(left.poses.is_empty());
            prop_assert!(right.poses.is_empty());
        }
        _ => prop_assert!(
            false,
            "{} changed whether station geometry produced renderable output bytes",
            what
        ),
    }
    prop_assert_eq!(
        left.poses.len(),
        right.poses.len(),
        "{} changed the number of solved sources",
        what
    );
    for (left_pose, right_pose) in left.poses.iter().zip(&right.poses) {
        prop_assert_eq!(
            &left_pose.0,
            &right_pose.0,
            "{} changed the solved source set",
            what
        );
        for (index, (left_value, right_value)) in
            left_pose.1.iter().zip(right_pose.1.iter()).enumerate()
        {
            prop_assert_eq!(
                left_value.to_bits(),
                right_value.to_bits(),
                "{} changed coefficient {} of the world pose of '{}' ({} vs {})",
                what,
                index,
                left_pose.0,
                left_value,
                right_value
            );
        }
    }
    Ok(())
}
