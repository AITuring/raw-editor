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
const SYNTHETIC_RANSAC_INLIER_THRESHOLD_PX: f64 = 2.0;

/// Minimum inliers demanded of a synthetic pairwise fit.
const SYNTHETIC_RANSAC_MIN_INLIERS: usize = 8;

/// Canvas samples reduced per placed station.
///
/// Above [`DETERMINISTIC_SUM_BLOCK_LEN`] on purpose, so the reduction takes the
/// blocked `rayon` path rather than the serial short-circuit — that is the path
/// whose independence from the thread count has to be observable.
const CANVAS_SAMPLES_PER_STATION: usize = 96 * 64;

/// Sample grid columns inside one station tile.
const CANVAS_SAMPLE_COLUMNS: usize = 96;

/// Geometry result of the production station solver.
///
/// A disconnected graph has no renderable pose/output result.  Its canonical
/// component member sets are retained so rename/import permutations can be
/// compared semantically instead of accidentally treating the old locked-pose
/// fallback as a successful solve.
#[derive(Debug, Clone, PartialEq, Eq)]
enum SyntheticGeometryOutcome {
    Connected,
    RejectedDisconnected {
        components: BTreeSet<BTreeSet<(usize, usize)>>,
        member_counts: Vec<usize>,
    },
}

fn synthetic_geometry_outcome(
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
struct SyntheticRun {
    /// Canonical bytes of every successful output decision plus every pixel
    /// aligned quantity it produced. Disconnected runs have no output bytes.
    placement: Option<Vec<u8>>,
    /// Solved world homography per source, keyed by file name so that the
    /// encoding cannot inherit the import order. Rejected runs keep this empty.
    poses: Vec<(String, Matrix3<f64>)>,
    geometry: SyntheticGeometryOutcome,
    /// Coverage witnesses: without them a degenerate run (fewer than two
    /// stations, no accepted relation) would satisfy every assertion below
    /// without ever entering the code under test.
    stations: usize,
    station_relations: usize,
    bracket_relations: usize,
}

fn map_point(transform: &Matrix3<f64>, point: Point2<f64>) -> Option<Point2<f64>> {
    let mapped = transform * Point3::new(point.x, point.y, 1.0);
    (mapped.z.abs() > 1e-8).then(|| Point2::new(mapped.x / mapped.z, mapped.y / mapped.z))
}

fn push_f64(bytes: &mut Vec<u8>, value: f64) {
    bytes.extend_from_slice(&value.to_bits().to_be_bytes());
}

fn push_matrix(bytes: &mut Vec<u8>, matrix: &Matrix3<f64>) {
    for value in matrix.iter() {
        push_f64(bytes, *value);
    }
}

fn push_str(bytes: &mut Vec<u8>, text: &str) {
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
fn determinism_images(scan: &SyntheticScan) -> Vec<ImageInfo> {
    let mut images = scan.images();
    for (image, &(station, layer)) in images.iter_mut().zip(scan.capture_identities().iter()) {
        let mut descriptor = [0u8; crate::panorama_stitching::BRIEF_DESCRIPTOR_SIZE / 8];
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
fn fit_synthetic_matches(
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
fn station_canvas_samples(
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

fn run_synthetic_stack(scan: &SyntheticScan) -> SyntheticRun {
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
        let canvas_reduction = super::determinism::deterministic_sum(&samples);

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
const THREAD_DIMENSION: [usize; 4] = [1, 2, 8, 16];

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
static THREAD_DIMENSION_CURSOR: AtomicUsize = AtomicUsize::new(0);

/// Bitmask of the [`THREAD_DIMENSION`] entries the accepted cases have run.
static THREAD_DIMENSION_COVERED: AtomicUsize = AtomicUsize::new(0);

/// Fixed size `rayon` pools, built once and reused across cases.
///
/// Building a pool spawns and parks its workers, which is pure overhead repeated
/// 100 times per worker count otherwise. A pool is stateless between
/// `install` calls, so reuse cannot carry anything from one case to the next.
fn shared_pool(threads: usize) -> &'static rayon::ThreadPool {
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
fn run_synthetic_stack_on_threads(scan: &SyntheticScan, threads: usize) -> SyntheticRun {
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
fn assert_runs_identical(
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

// ---------------------------------------------------------------------------
// Property 2 harness: station membership and the station level topology
// ---------------------------------------------------------------------------
//
// Property 90 above asserts that the *whole* placement byte string is
// reproducible, but only across a repeated run, the thread count and an import
// permutation.  Property 2 covers the two quantities that sit earlier in the
// pipeline — the Capture_Station membership partition and the station level
// topology — and adds the dimension Property 90 does not touch at all: a
// **rename** of the whole selection (需求 1.3).
//
// The harness drives the same production functions Property 90 does, in the
// same order, but stops at the topology instead of the canvas reduction:
//
//   1. `canonical_match_direction` mirrored, then
//      `processing::find_homography_ransac_points_stable_with_min_inliers` —
//      the real pairwise RANSAC.
//   2. `solve_focus_capture_group_poses` — the station pose solve.
//   3. `focus_capture_groups` — the station membership partition *and* the
//      station order, which is its world-centre sort.
//   4. `focus_station_topology` plus
//      `focus_source_order_group_compositor_order` — production row/column
//      inference and the default layered compositor order.
//
// The row/column mapping and Candidate_Adjacency set are covered directly by
// the topology module tests from tasks 9.1/9.2. This earlier property expresses
// compositor order as station member identities, so rename and shuffled import
// indices cannot make the comparison circular.

/// Capture identity of a source: its station and its focus rank inside it.
///
/// Members must be compared by *content*, never by file name (需求 1.3) and
/// never by import index, or the comparison would be circular.
type CaptureIdentity = (usize, usize);

/// One Capture_Station, as the set of capture identities it owns.
type StationMembers = BTreeSet<CaptureIdentity>;

/// Station membership and station level topology of one synthetic run.
#[derive(Debug, PartialEq, Eq)]
struct GroupingTopology {
    /// Station member sets. Their partition exists before pose rejection.
    stations: Vec<StationMembers>,
    /// Connected runs carry compositor order; disconnected runs carry the
    /// canonical explicit rejection report and intentionally carry no order.
    geometry: SyntheticGeometryOutcome,
    /// `focus_source_order_group_compositor_order`, expressed as the sequence
    /// of station member sets rather than of station indices, so the assertion
    /// survives a future renumbering of the stations.
    compositor_order: Vec<StationMembers>,
}

impl GroupingTopology {
    /// The membership partition alone — the quantity 需求 1.3 names.
    ///
    /// Order carrying comparisons use the fields directly; this drops the
    /// station order so the partition can be asserted on its own.
    fn partition(&self) -> BTreeSet<StationMembers> {
        self.stations.iter().cloned().collect()
    }
}

/// Coverage witnesses: without them a degenerate scan (one station, no accepted
/// relation) would satisfy every assertion below without entering the grouper.
#[derive(Debug)]
struct GroupingWitness {
    station_relations: usize,
    bracket_relations: usize,
}

fn grouping_topology(scan: &SyntheticScan) -> (GroupingTopology, GroupingWitness) {
    let images = determinism_images(scan);
    let locked = scan.locked_homographies();
    let identities = scan.capture_identities();

    // 1. Real pairwise RANSAC over the correspondences of the scan.
    let (matches, _measured) = fit_synthetic_matches(scan, &images);
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

    // 2. The station pose solve. Disconnected geometry is retained as an
    // explicit canonical rejection report, never as locked output poses.
    let solve = determinism_test_access::solve_station_poses_with_report(
        &images,
        &matches,
        &locked,
        scan.geometric_reference_index(),
    );
    let geometry = synthetic_geometry_outcome(&images, &identities, &solve);

    // 3. Grouping is a pre-solve operation in production. Re-running it from
    // solved station poses can renumber otherwise identical Capture_Stations
    // when an allowed rename changes the near-tie anchor frame. Keep station
    // identity tied to the same locked image-level evidence the production
    // solver grouped, then use solved poses only for topology/composition.
    let stations =
        determinism_test_access::capture_stations(&images, &matches, &locked).unwrap_or_default();
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

    let member_identities = |station: &determinism_test_access::StationView| {
        station
            .members
            .iter()
            .filter_map(|&index| identities.get(index).copied())
            .collect::<StationMembers>()
    };

    // 4. The compositor / seam order from the same production topology
    //    inference used by the default layered virtual-tile compositor.
    let compositor_order =
        if geometry == SyntheticGeometryOutcome::Connected && geometries.len() == stations.len() {
            determinism_test_access::station_compositor_order(&images, &stations, &solve.poses)
        } else {
            Vec::new()
        };

    (
        GroupingTopology {
            stations: stations.iter().map(member_identities).collect(),
            geometry,
            compositor_order: compositor_order
                .iter()
                .filter_map(|&station| stations.get(station))
                .map(member_identities)
                .collect(),
        },
        GroupingWitness {
            station_relations,
            bracket_relations,
        },
    )
}

/// Assert that `candidate` holds the same Capture_Station membership partition
/// as `reference` (需求 1.3), ignoring the station order.
fn assert_partition_identical(
    reference: &GroupingTopology,
    candidate: &GroupingTopology,
    what: &str,
) -> Result<(), TestCaseError> {
    prop_assert_eq!(
        reference.stations.len(),
        candidate.stations.len(),
        "{} changed the Capture_Station count",
        what
    );
    prop_assert_eq!(
        reference.partition(),
        candidate.partition(),
        "{} changed the Capture_Station membership partition",
        what
    );
    Ok(())
}

/// Assert the partition *and* the station level topology (需求 5.8).
fn assert_topology_identical(
    reference: &GroupingTopology,
    candidate: &GroupingTopology,
    what: &str,
) -> Result<(), TestCaseError> {
    assert_partition_identical(reference, candidate, what)?;
    prop_assert_eq!(
        &reference.geometry,
        &candidate.geometry,
        "{} changed the connected/rejected outcome or canonical disconnected-component report",
        what
    );
    match &reference.geometry {
        SyntheticGeometryOutcome::Connected => {
            for (station, (expected, actual)) in reference
                .stations
                .iter()
                .zip(&candidate.stations)
                .enumerate()
            {
                prop_assert_eq!(
                    expected,
                    actual,
                    "{} changed the members of station {} in canonical topology order",
                    what,
                    station
                );
            }
            prop_assert_eq!(
                &reference.compositor_order,
                &candidate.compositor_order,
                "{} changed the compositor order",
                what
            );
        }
        SyntheticGeometryOutcome::RejectedDisconnected { .. } => {
            prop_assert!(reference.compositor_order.is_empty());
            prop_assert!(candidate.compositor_order.is_empty());
        }
    }
    Ok(())
}

/// How far each rename scheme is asserted.
///
/// Determinism fixtures carry immutable feature signatures and generate
/// correspondence support in the same content-derived direction as production
/// decoded images. Therefore every rename scheme is held to the full grouping,
/// accepted/rejected geometry outcome, canonical disconnected-component report,
/// and connected topology contract; no filename-fallback exception remains.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Strictness {
    /// Partition only (需求 1.3).
    Partition,
    /// Partition plus the station order and the compositor order (需求 5.8).
    Topology,
}

/// The rename dimension of Property 2, with the strictness each scheme is held
/// to and the reason for anything short of [`Strictness::Topology`].
const RENAME_DIMENSION: [(RenameScheme, Strictness); 4] = [
    // Name order and numeric gaps untouched: nothing the pipeline is allowed to
    // read changed, so everything must hold.
    (RenameScheme::ForeignPrefix, Strictness::Topology),
    (RenameScheme::ShiftedNumberBase, Strictness::Topology),
    // Name order untouched but every `trailing_capture_number` gap moved above
    // `FOCUS_BRACKET_MAX_CAPTURE_GAP`, so the pose graph weight boosts and the
    // low-texture recovery paths that still key on that gap lose their bonus.
    // 需求 1.2 forbids the file name from deciding anything beyond a candidate
    // order and a ≤0.001 tie-break, so this is held to the full topology: if it
    // ever fails, a filename-number dependency has become load bearing rather
    // than the test having become flaky.
    (RenameScheme::WideNumberStride, Strictness::Topology),
    // Name order inverted: immutable feature signatures still choose the same
    // asymmetric measurement direction, so full topology invariance applies.
    (RenameScheme::ReversedNumbering, Strictness::Topology),
];

/// Number of distinct random import orders exercised, per 需求 1.3
/// ("至少 5 种不同随机顺序").
const IMPORT_PERMUTATION_ROUNDS: u64 = 5;

/// Seed of import permutation `round`.
fn import_permutation_seed(round: u64) -> u64 {
    0x9E37_79B9_7F4A_7C15u64
        .wrapping_mul(round + 1)
        .wrapping_add(round)
}

/// One transform Property 2 holds the grouper invariant under.
#[derive(Clone, Copy, Debug)]
enum NameOrderTransform {
    /// A renaming scheme, by index into [`RENAME_DIMENSION`].
    Rename(usize),
    /// An import permutation, by round.
    Permutation(u64),
    /// A renaming scheme with an import permutation on top, which is what "the
    /// user copied the files, renamed them and dragged them in again" looks like.
    RenameAndPermutation(usize),
}

/// The 13 transforms of Property 2: four renaming schemes, five import
/// permutations, and each renaming scheme once more with a permutation on top.
const NAME_ORDER_DIMENSION: usize = 2 * RENAME_DIMENSION.len() + IMPORT_PERMUTATION_ROUNDS as usize;

fn name_order_transform(slot: usize) -> NameOrderTransform {
    if slot < RENAME_DIMENSION.len() {
        NameOrderTransform::Rename(slot)
    } else if slot < RENAME_DIMENSION.len() + IMPORT_PERMUTATION_ROUNDS as usize {
        NameOrderTransform::Permutation((slot - RENAME_DIMENSION.len()) as u64)
    } else {
        NameOrderTransform::RenameAndPermutation(
            slot - RENAME_DIMENSION.len() - IMPORT_PERMUTATION_ROUNDS as usize,
        )
    }
}

/// Number of accepted Property 2 cases so far, which drives the rotation through
/// [`NAME_ORDER_DIMENSION`].
///
/// Running all thirteen transforms on every case costs fourteen full grouping
/// passes per case and buys nothing: the assertion is that the grouping is
/// independent of names and arrival order, and a counterexample would have to
/// appear for *some* input under *some* transform. Rotating one transform per
/// case keeps the whole dimension covered across the run (asserted below) at a
/// seventh of the cost, and `proptest` executes the cases of one test
/// sequentially, so the rotation is a deterministic function of the accepted
/// case index.
static NAME_ORDER_CURSOR: AtomicUsize = AtomicUsize::new(0);

/// Bitmask of the [`NAME_ORDER_DIMENSION`] slots the accepted cases have run.
static NAME_ORDER_COVERED: AtomicUsize = AtomicUsize::new(0);

fn assert_invariance(
    reference: &GroupingTopology,
    candidate: &GroupingTopology,
    strictness: Strictness,
    what: &str,
) -> Result<(), TestCaseError> {
    match strictness {
        Strictness::Partition => {
            assert_partition_identical(reference, candidate, what)?;
            prop_assert_eq!(
                &reference.geometry,
                &candidate.geometry,
                "{} changed the connected/rejected outcome or canonical disconnected-component report",
                what
            );
            Ok(())
        }
        Strictness::Topology => assert_topology_identical(reference, candidate, what),
    }
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 100, ..ProptestConfig::default() })]

    // Feature: layered-camera-group-focus-stitching, Property 2: 对于任意 Source_RAW
    // 集合与任意重命名或导入顺序置换，Station_Grouper 输出的 Capture_Station 成员集合划分、
    // Capture_Topology_Model 输出的行列索引映射与 Candidate_Adjacency 集合都完全相同。
    //
    // 文件名数字与导入顺序只允许用于生成候选顺序，以及只允许在图像证据判定分数之差不超过
    // 0.001 时作为稳定 tie-break（需求 1.2）；重命名或至少 5 种不同随机导入顺序必须给出
    // 完全相同的机位成员集合划分（需求 1.3）与相同的拓扑（需求 5.8）。
    //
    // **Validates: Requirements 1.2, 1.3, 5.8**
    #[test]
    fn grouping_and_topology_are_invariant_under_rename_and_import_order(
        scan in arb_artwork_scan_grid()
    ) {
        let _run_scope = degradation::begin_run_scope();
        prop_assume!(scan.station_count() >= 2);

        let (reference, witness) = grouping_topology(&scan);

        // Coverage: the grouper is only under test when it really had several
        // stations, focus bracket links and inter-station links to work with.
        prop_assert!(
            reference.stations.len() >= 2,
            "the harness must reach a multi station grouping, got {} station(s)",
            reference.stations.len()
        );
        prop_assert!(
            witness.bracket_relations >= 1,
            "the harness must accept at least one focus bracket relation"
        );
        prop_assert!(
            witness.station_relations >= 1,
            "the harness must accept at least one inter-station relation"
        );
        match &reference.geometry {
            SyntheticGeometryOutcome::Connected => prop_assert_eq!(
                reference.compositor_order.len(),
                reference.stations.len(),
                "every station in connected geometry must appear exactly once in compositor order"
            ),
            SyntheticGeometryOutcome::RejectedDisconnected { components, .. } => {
                prop_assert!(components.len() >= 2);
                prop_assert!(reference.compositor_order.is_empty());
            }
        }

        // Every persisted scan must exercise the historically failing reverse
        // numbering directly. A process-global transform cursor cannot make a
        // saved counterexample replay under a different rename scheme.
        let reversed = scan.with_renamed_sources(RenameScheme::ReversedNumbering);
        prop_assert_eq!(
            reversed.capture_identities(),
            scan.capture_identities(),
            "reversed numbering must leave capture identities untouched"
        );
        let (reversed_candidate, _) = grouping_topology(&reversed);
        assert_topology_identical(
            &reference,
            &reversed_candidate,
            "rename ReversedNumbering",
        )?;

        // One of the thirteen additional transforms per accepted case, rotating through
        // `NAME_ORDER_DIMENSION`; with `cases: 100` every transform runs at least
        // seven times, and the assertion at the end turns "the rotation really
        // covered all thirteen" into a test failure rather than a comment.
        let case = NAME_ORDER_CURSOR.fetch_add(1, Ordering::Relaxed);
        let slot = case % NAME_ORDER_DIMENSION;
        let covered = NAME_ORDER_COVERED.fetch_or(1 << slot, Ordering::Relaxed) | (1 << slot);
        match name_order_transform(slot) {
            // 重命名不变性 (需求 1.3): another camera's prefix, another numbering
            // base, a numbering stride above the capture-gap limit and a
            // numbering that runs against the geometric scan order.
            NameOrderTransform::Rename(index) => {
                let (scheme, strictness) = RENAME_DIMENSION[index];
                let renamed = scan.with_renamed_sources(scheme);
                prop_assert_ne!(
                    renamed.filenames(),
                    scan.filenames(),
                    "{:?} must actually rename the selection",
                    scheme
                );
                prop_assert_eq!(
                    renamed.capture_identities(),
                    scan.capture_identities(),
                    "{:?} must leave the capture identities untouched",
                    scheme
                );
                let (candidate, _) = grouping_topology(&renamed);
                assert_invariance(
                    &reference,
                    &candidate,
                    strictness,
                    &format!("rename {scheme:?}"),
                )?;
            }
            // 导入顺序不变性 (需求 1.3 asks for at least 5 different random
            // orders).  Each round uses a different seed, and a drawn identity
            // permutation is replaced by the reversal, so none of the five can
            // pass vacuously.  This dimension is asserted at full strictness:
            // an import permutation changes no file name, so no production site
            // has any excuse to react to it.
            NameOrderTransform::Permutation(round) => {
                let permuted =
                    scan.with_permuted_import_order_seeded(import_permutation_seed(round));
                prop_assert_ne!(
                    permuted.capture_identities(),
                    scan.capture_identities(),
                    "permutation {} must actually reorder the import",
                    round
                );
                let (candidate, _) = grouping_topology(&permuted);
                assert_topology_identical(
                    &reference,
                    &candidate,
                    &format!("import permutation {round}"),
                )?;
            }
            // Both transforms at once.  The strictness of the rename scheme
            // carries over.
            NameOrderTransform::RenameAndPermutation(index) => {
                let (scheme, strictness) = RENAME_DIMENSION[index];
                let both = scan
                    .with_renamed_sources(scheme)
                    .with_permuted_import_order_seeded(import_permutation_seed(
                        IMPORT_PERMUTATION_ROUNDS,
                    ));
                let (candidate, _) = grouping_topology(&both);
                assert_invariance(
                    &reference,
                    &candidate,
                    strictness,
                    &format!("rename {scheme:?} plus an import permutation"),
                )?;
            }
        }
        prop_assert!(
            case + 1 < NAME_ORDER_DIMENSION || covered == (1usize << NAME_ORDER_DIMENSION) - 1,
            "case {} must have covered all {} rename/import transforms by now, covered mask {:#b}",
            case,
            NAME_ORDER_DIMENSION,
            covered
        );
    }
}
