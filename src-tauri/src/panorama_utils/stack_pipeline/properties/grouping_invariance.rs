//! Property 2: station membership and the station level topology are invariant
//! under rename and import order.
//!
//! Members are compared by capture identity, never by file name or import index.

use super::super::degradation;
use super::super::test_support::RenameScheme;
use super::super::test_support::SyntheticScan;
use super::super::test_support::arb_artwork_scan_grid;
use crate::panorama_stitching::determinism_test_access;
use crate::panorama_utils::stitching;
use proptest::prelude::*;
use std::collections::BTreeSet;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

use super::{
    SyntheticGeometryOutcome, determinism_images, fit_synthetic_matches, synthetic_geometry_outcome,
};

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
pub(super) type CaptureIdentity = (usize, usize);

/// One Capture_Station, as the set of capture identities it owns.
pub(super) type StationMembers = BTreeSet<CaptureIdentity>;

/// Station membership and station level topology of one synthetic run.
#[derive(Debug, PartialEq, Eq)]
pub(super) struct GroupingTopology {
    /// Station member sets. Their partition exists before pose rejection.
    pub(super) stations: Vec<StationMembers>,
    /// Connected runs carry compositor order; disconnected runs carry the
    /// canonical explicit rejection report and intentionally carry no order.
    pub(super) geometry: SyntheticGeometryOutcome,
    /// `focus_source_order_group_compositor_order`, expressed as the sequence
    /// of station member sets rather than of station indices, so the assertion
    /// survives a future renumbering of the stations.
    pub(super) compositor_order: Vec<StationMembers>,
}

impl GroupingTopology {
    /// The membership partition alone — the quantity 需求 1.3 names.
    ///
    /// Order carrying comparisons use the fields directly; this drops the
    /// station order so the partition can be asserted on its own.
    pub(super) fn partition(&self) -> BTreeSet<StationMembers> {
        self.stations.iter().cloned().collect()
    }
}

/// Coverage witnesses: without them a degenerate scan (one station, no accepted
/// relation) would satisfy every assertion below without entering the grouper.
#[derive(Debug)]
pub(super) struct GroupingWitness {
    pub(super) station_relations: usize,
    pub(super) bracket_relations: usize,
}

pub(super) fn grouping_topology(scan: &SyntheticScan) -> (GroupingTopology, GroupingWitness) {
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
pub(super) fn assert_partition_identical(
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
pub(super) fn assert_topology_identical(
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
pub(super) enum Strictness {
    /// Partition only (需求 1.3).
    Partition,
    /// Partition plus the station order and the compositor order (需求 5.8).
    Topology,
}

/// The rename dimension of Property 2, with the strictness each scheme is held
/// to and the reason for anything short of [`Strictness::Topology`].
pub(super) const RENAME_DIMENSION: [(RenameScheme, Strictness); 4] = [
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
pub(super) const IMPORT_PERMUTATION_ROUNDS: u64 = 5;

/// Seed of import permutation `round`.
pub(super) fn import_permutation_seed(round: u64) -> u64 {
    0x9E37_79B9_7F4A_7C15u64
        .wrapping_mul(round + 1)
        .wrapping_add(round)
}

/// One transform Property 2 holds the grouper invariant under.
#[derive(Clone, Copy, Debug)]
pub(super) enum NameOrderTransform {
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
pub(super) const NAME_ORDER_DIMENSION: usize =
    2 * RENAME_DIMENSION.len() + IMPORT_PERMUTATION_ROUNDS as usize;

pub(super) fn name_order_transform(slot: usize) -> NameOrderTransform {
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
pub(super) static NAME_ORDER_CURSOR: AtomicUsize = AtomicUsize::new(0);

/// Bitmask of the [`NAME_ORDER_DIMENSION`] slots the accepted cases have run.
pub(super) static NAME_ORDER_COVERED: AtomicUsize = AtomicUsize::new(0);

pub(super) fn assert_invariance(
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
