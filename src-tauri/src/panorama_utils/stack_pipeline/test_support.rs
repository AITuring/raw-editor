//! Shared `proptest` generators for the Stack_Pipeline property tests.
//!
//! The generators live in one module so every property task of the rollout
//! reuses the same inputs instead of inventing a private shape per test.  This
//! file is compiled only under `cfg(test)`; nothing in here is reachable from a
//! production path.
//!
//! Design reference: `Testing Strategy > 生成器设计` of the
//! `layered-camera-group-focus-stitching` design document.  This task adds
//! `arb_stack_report()`, the "random but schema legal Stack_Report, including
//! out-of-range combinations" generator.

// Later property tasks consume the remaining generators; keep them public
// inside the test build without warning noise in the meantime.
#![allow(dead_code)]

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use image::{GrayImage, Luma, Rgb, Rgb32FImage};
use nalgebra::{Matrix3, Point2, Point3};
use proptest::prelude::*;
use serde_json::{Value, json};

use crate::panorama_stitching::{ImageInfo, MatchInfo};
use crate::panorama_utils::registration;
use crate::panorama_utils::stitching::{
    ColorEncoding, ConfidenceMap, CoverageMask, OwnershipMap, SourceProvenance, VirtualTile,
};

use super::degradation::{DegradationLedger, FAILURE_REASONS, Severity, UNMEASURABLE_REASONS};
use super::intra_station::AnchorCandidate;
use super::report::{
    CanvasRecord, ClosureReport, ClosureStatus, CompositionReport, DegradationEntryRecord,
    DegradationReport, InputReport, QualityGateCriterionRecord, QualityGateReport,
    QualityGateVerdict, SelectedPath, SourceRecord, StackReport, StackRunResult,
    UnmeasurableRecord, WorldPoint,
};

/// The stable `quality_gate.criteria[*].name` identifier set of the design
/// document.  Used so a generated report carries realistic criterion names.
pub const QUALITY_GATE_CRITERION_NAMES: &[&str] = &[
    "local_scale_median",
    "local_scale_pixel_ratio",
    "effective_pixel_count",
    "mtf50_normalized",
    "gradient_energy_normalized",
    "noise_sigma_ratio",
    "roi_delta_e00",
    "boundary_stroke_alignment",
    "sharpness_confidence_coverage",
];

/// A failure reason identifier together with the severity the design matrix
/// declares for it.
pub fn arb_failure_reason() -> impl Strategy<Value = (&'static str, Severity)> {
    prop::sample::select(FAILURE_REASONS.to_vec())
}

/// An unmeasurable reason identifier (requirement 11.13).
pub fn arb_unmeasurable_reason() -> impl Strategy<Value = &'static str> {
    prop::sample::select(UNMEASURABLE_REASONS.to_vec())
}

fn arb_criterion_name() -> impl Strategy<Value = &'static str> {
    prop::sample::select(QUALITY_GATE_CRITERION_NAMES.to_vec())
}

/// Free-form measured context of a degradation entry.  Values deliberately run
/// past the thresholds of the design document so a report can carry
/// out-of-range combinations.
pub fn arb_degradation_detail() -> impl Strategy<Value = Value> {
    prop_oneof![
        Just(Value::Null),
        (-1.0e6f64..1.0e6f64).prop_map(|measured| json!({ "measured": measured })),
        (0.0f64..1.0e4f64, 0.0f64..1.0e4f64).prop_map(
            |(threshold, measured)| json!({ "threshold": threshold, "measured": measured })
        ),
        (0usize..512, 0usize..512)
            .prop_map(|(left, right)| json!({ "left": left, "right": right })),
        any::<bool>().prop_map(|clamped| json!({ "clamped": clamped })),
        (0usize..1024).prop_map(|count| json!({ "count": count })),
    ]
}

/// A ledger holding 0..8 active degraded or rejected paths.
pub fn arb_degradation_ledger() -> impl Strategy<Value = DegradationLedger> {
    prop::collection::vec((arb_failure_reason(), arb_degradation_detail()), 0..8).prop_map(
        |records| {
            let mut ledger = DegradationLedger::new();
            for ((reason, _severity), detail) in records {
                ledger.record(reason, detail);
            }
            ledger
        },
    )
}

fn degradation_entry_records(ledger: &DegradationLedger) -> Vec<DegradationEntryRecord> {
    ledger
        .entries()
        .iter()
        .map(|entry| DegradationEntryRecord {
            reason: entry.reason.to_string(),
            severity: entry.severity.as_identifier().to_string(),
            detail: entry.detail.clone(),
        })
        .collect()
}

fn arb_selected_path() -> impl Strategy<Value = SelectedPath> {
    prop_oneof![
        Just(SelectedPath::LayeredVirtualTile),
        Just(SelectedPath::ProgressiveSeamTile),
        Just(SelectedPath::StreamingMosaic),
        Just(SelectedPath::LegacySingleLayerMosaic),
        Just(SelectedPath::SingleStation),
    ]
}

fn arb_run_result() -> impl Strategy<Value = StackRunResult> {
    prop_oneof![
        Just(StackRunResult::Running),
        Just(StackRunResult::Success),
        Just(StackRunResult::Degraded),
        Just(StackRunResult::Rejected),
        Just(StackRunResult::Cancelled),
    ]
}

fn arb_quality_gate_verdict() -> impl Strategy<Value = QualityGateVerdict> {
    prop_oneof![
        Just(QualityGateVerdict::Pass),
        Just(QualityGateVerdict::Fail),
        Just(QualityGateVerdict::InsufficientEvidence),
        Just(QualityGateVerdict::NotRun),
    ]
}

fn arb_closure_status() -> impl Strategy<Value = ClosureStatus> {
    prop_oneof![
        Just(ClosureStatus::Converged),
        Just(ClosureStatus::Unreliable),
        Just(ClosureStatus::NoConstraints),
        Just(ClosureStatus::NotRun),
    ]
}

/// Source records with plausible absolute paths and digests.  The digests are
/// synthetic hex strings: no property in this stage recomputes them.
fn arb_input_report() -> impl Strategy<Value = InputReport> {
    // `source_count` is free to exceed `max_sources` so the out-of-range
    // rejection combination is representable.
    (0usize..600, prop::collection::vec(0u8..16, 0..6)).prop_map(|(source_count, digits)| {
        let sources = digits
            .iter()
            .enumerate()
            .map(|(index, digit)| {
                let digest =
                    std::iter::repeat_n(char::from_digit(u32::from(*digit), 16).unwrap_or('0'), 64)
                        .collect::<String>();
                SourceRecord {
                    path: format!("/tmp/stack-pipeline/DSC_{:04}.NEF", 3680 + index),
                    sha256_before: digest.clone(),
                    sha256_after: digest,
                    decoded: index % 3 != 0,
                }
            })
            .collect();
        InputReport {
            source_count,
            max_sources: 500,
            sources,
        }
    })
}

/// Canvas sizes straddle the 262,144 world-pixel long-side ceiling.
fn arb_composition_report() -> impl Strategy<Value = CompositionReport> {
    (1u64..400_000, 1u64..400_000, 0u64..1_000_000).prop_map(|(width, height, opaque_pixels)| {
        CompositionReport {
            canvas: CanvasRecord {
                width,
                height,
                ..CanvasRecord::default()
            },
            opaque_pixels,
            ..CompositionReport::default()
        }
    })
}

/// Closure residuals straddle the 2.0 / 3.0 world-pixel limits and the 100
/// iteration ceiling.
fn arb_closure_report() -> impl Strategy<Value = ClosureReport> {
    (
        arb_closure_status(),
        0usize..200,
        0.0f64..10.0,
        0.0f64..20.0,
        0.0f64..20.0,
        0.0f64..1024.0,
    )
        .prop_map(
            |(
                status,
                iterations,
                residual_median_px,
                residual_p95_px,
                max_direct_pair_p95_px,
                max_corner_correction_px,
            )| ClosureReport {
                status,
                iterations,
                residual_median_px,
                residual_p95_px,
                max_direct_pair_p95_px,
                max_corner_correction_px,
                ..ClosureReport::default()
            },
        )
}

fn arb_quality_gate_report() -> impl Strategy<Value = QualityGateReport> {
    (
        arb_quality_gate_verdict(),
        0usize..64,
        prop::collection::vec(
            (arb_criterion_name(), 0.0f64..4.0, 0usize..32, 0usize..32),
            0..4,
        ),
        prop::collection::vec(
            (
                arb_criterion_name(),
                arb_unmeasurable_reason(),
                -1.0e5f64..1.0e5f64,
                -1.0e5f64..1.0e5f64,
            ),
            0..6,
        ),
    )
        .prop_map(
            |(verdict, roi_count, criteria, unmeasurable)| QualityGateReport {
                verdict,
                roi_count,
                criteria: criteria
                    .into_iter()
                    .map(|(name, threshold, measurable_count, unmeasurable_count)| {
                        QualityGateCriterionRecord {
                            name: name.to_string(),
                            threshold,
                            measurable_count,
                            unmeasurable_count,
                            ..QualityGateCriterionRecord::default()
                        }
                    })
                    .collect(),
                unmeasurable: unmeasurable
                    .into_iter()
                    .map(|(criterion, reason, x, y)| UnmeasurableRecord {
                        criterion: criterion.to_string(),
                        world: WorldPoint { x, y },
                        reason: reason.to_string(),
                    })
                    .collect(),
            },
        )
}

/// A random but schema-legal [`StackReport`].
///
/// Every failure reason identifier the document carries is drawn from the
/// frozen identifier sets of `degradation.rs`, which is exactly the invariant
/// Property 79 relies on: the pipeline can only ever write identifiers from
/// that set.  Numeric fields intentionally cross the thresholds of the design
/// document so out-of-range combinations are representable.
pub fn arb_stack_report() -> impl Strategy<Value = StackReport> {
    (
        arb_degradation_ledger(),
        arb_run_result(),
        arb_selected_path(),
        arb_input_report(),
        arb_composition_report(),
        arb_closure_report(),
        arb_quality_gate_report(),
        0u64..1_000_000,
    )
        .prop_map(
            |(
                ledger,
                result,
                selected_path,
                input,
                composition,
                closure,
                quality_gate,
                started_at_epoch_ms,
            )| {
                StackReport {
                    pipeline_version: "stack-pipeline-test".to_string(),
                    run_id: format!("run-{started_at_epoch_ms}"),
                    started_at_epoch_ms,
                    finished_at_epoch_ms: started_at_epoch_ms,
                    selected_path,
                    input,
                    composition,
                    closure,
                    quality_gate,
                    degradation: DegradationReport {
                        entries: degradation_entry_records(&ledger),
                        result,
                    },
                    ..StackReport::default()
                }
            },
        )
}

// ---------------------------------------------------------------------------
// Synthetic scan generators (设计 `Testing Strategy > 生成器设计`)
// ---------------------------------------------------------------------------
//
// `arb_scan_grid(plane)` produces a serpentine two dimensional scan over a
// procedurally generated "artwork" plane: 2..6 capture stations with 15%..40%
// overlap, per-station drift and per-station exposure differences, each station
// holding 2..3 focus layers.
//
// Everything is derived from the plane by *cropping through a known pose*, so
// the correspondences handed to the registration code are geometric ground
// truth rather than invented numbers, and the tile pixels are real image
// content the dense overlap verification can correlate.

/// Side of the procedurally generated artwork plane, in plane pixels.
pub const SYNTHETIC_PLANE_SIDE: u32 = 640;

/// Plane coordinate of the top-left corner of the scanned area.
const SYNTHETIC_SCAN_ORIGIN: f64 = 64.0;

/// Side of one synthetic capture tile, in tile pixels.
///
/// Deliberately above the 32 px floor `focus_overlap_quality` needs, so that
/// the dense overlap verification inside the station solver really runs instead
/// of being skipped as unmeasurable.
pub const SYNTHETIC_TILE_SIDE: u32 = 160;

/// Feature correspondences kept per overlapping pair.
///
/// A real match set is bounded; keeping it bounded here also keeps the RANSAC
/// inlier evaluation of the property test affordable.
///
/// The cap must stay above the 30 verified inliers 需求 1.1 requires of a
/// Capture_Station link (`STATION_MIN_INLIER_MATCHES`), not merely above the 24
/// of a 需求 6.3 station relation: a fixture that hands the grouper 24
/// correspondences per focus-bracket pair can never produce an accepted
/// membership relation, so the harness would exercise the fallbacks instead of
/// the grouper itself.
const SYNTHETIC_MAX_CORRESPONDENCES: usize = 48;

/// Grid resolution used to search for correspondences inside a tile.
const SYNTHETIC_CORRESPONDENCE_GRID: u32 = 12;

/// `SplitMix64`: a tiny, fully specified bit mixer.
///
/// Used instead of a crate RNG so that the generated plane is a pure function
/// of the `proptest` drawn seed on every platform.
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut mixed = *state;
    mixed = (mixed ^ (mixed >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    mixed = (mixed ^ (mixed >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    mixed ^ (mixed >> 31)
}

/// Uniform value in `[0, 1)`.
fn unit_interval(state: &mut u64) -> f64 {
    (splitmix64(state) >> 11) as f64 / (1u64 << 53) as f64
}

/// Uniform value in `[low, high)`.
fn in_range(state: &mut u64, low: f64, high: f64) -> f64 {
    low + unit_interval(state) * (high - low)
}

/// Apply a homography to a point; `None` when the result is not finite.
fn map_point(transform: &Matrix3<f64>, point: Point2<f64>) -> Option<Point2<f64>> {
    let mapped = transform * Point3::new(point.x, point.y, 1.0);
    (mapped.z.abs() > 1e-8).then(|| Point2::new(mapped.x / mapped.z, mapped.y / mapped.z))
}

/// The synthetic "书画"底图: paper grain plus a handful of variable width
/// Bézier strokes and large empty areas.
#[derive(Clone)]
pub struct ArtworkPlane {
    image: Arc<GrayImage>,
}

impl ArtworkPlane {
    pub fn image(&self) -> &GrayImage {
        &self.image
    }
}

/// A summary rather than 400 KiB of pixels: `proptest` prints this on failure.
impl std::fmt::Debug for ArtworkPlane {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let (width, height) = self.image.dimensions();
        let mean = self
            .image
            .as_raw()
            .iter()
            .map(|&value| u64::from(value))
            .sum::<u64>()
            / u64::from(width * height).max(1);
        formatter
            .debug_struct("ArtworkPlane")
            .field("width", &width)
            .field("height", &height)
            .field("mean_luma", &mean)
            .finish()
    }
}

fn paint_stroke_sample(image: &mut GrayImage, x: f64, y: f64, radius: f64, ink: u8) {
    let (width, height) = image.dimensions();
    let min_x = (x - radius).floor().max(0.0) as u32;
    let min_y = (y - radius).floor().max(0.0) as u32;
    let max_x = ((x + radius).ceil() as i64).clamp(0, i64::from(width) - 1) as u32;
    let max_y = ((y + radius).ceil() as i64).clamp(0, i64::from(height) - 1) as u32;
    for pixel_y in min_y..=max_y {
        for pixel_x in min_x..=max_x {
            let dx = f64::from(pixel_x) + 0.5 - x;
            let dy = f64::from(pixel_y) + 0.5 - y;
            if dx * dx + dy * dy > radius * radius {
                continue;
            }
            let current = image.get_pixel(pixel_x, pixel_y)[0];
            image.put_pixel(pixel_x, pixel_y, Luma([current.min(ink)]));
        }
    }
}

fn artwork_plane(seed: u64, stroke_count: usize) -> ArtworkPlane {
    let side = SYNTHETIC_PLANE_SIDE;
    // Paper: a bright base with a fine grain, so even the 留白 carries enough
    // signal for the dense overlap correlation to be measurable.
    let mut image = GrayImage::from_fn(side, side, |x, y| {
        let mut local = seed
            ^ (u64::from(x).wrapping_mul(0x2545_F491_4F6C_DD1D))
            ^ (u64::from(y).wrapping_mul(0x9E37_79B9_7F4A_7C15));
        Luma([238u8.saturating_sub((splitmix64(&mut local) % 13) as u8)])
    });
    let mut state = seed | 1;
    for _ in 0..stroke_count {
        let control: [Point2<f64>; 3] = std::array::from_fn(|_| {
            Point2::new(
                in_range(&mut state, 0.0, f64::from(side)),
                in_range(&mut state, 0.0, f64::from(side)),
            )
        });
        let ink = in_range(&mut state, 12.0, 92.0) as u8;
        let half_width = in_range(&mut state, 1.2, 4.0);
        let steps = 320usize;
        for step in 0..=steps {
            let t = step as f64 / steps as f64;
            let inverse = 1.0 - t;
            let x = inverse * inverse * control[0].x
                + 2.0 * inverse * t * control[1].x
                + t * t * control[2].x;
            let y = inverse * inverse * control[0].y
                + 2.0 * inverse * t * control[1].y
                + t * t * control[2].y;
            // A brush tapers at both ends.
            let radius = half_width * (0.45 + 0.55 * (t * std::f64::consts::PI).sin());
            paint_stroke_sample(&mut image, x, y, radius, ink);
        }
    }
    ArtworkPlane {
        image: Arc::new(image),
    }
}

/// Random artwork plane: 6..14 strokes over a grained paper base.
pub fn arb_artwork_plane() -> impl Strategy<Value = ArtworkPlane> {
    (any::<u64>(), 6usize..14).prop_map(|(seed, strokes)| artwork_plane(seed, strokes))
}

/// A densely painted artwork plane whose grain is band limited.
///
/// The Capture_Station properties measure the 需求 1.1 overlap correlation between
/// two focus layers of one camera position, and those two layers are crops of the
/// plane through *different sub-pixel* poses.  Per-pixel white grain cannot
/// survive that: two different resamplings of it decorrelate, and on a sparsely
/// painted plane the grain is most of the signal, so the measured correlation of a
/// genuine intra-station pair drops below the 0.60 bar for reasons that have
/// nothing to do with the grouping.
///
/// A real analysis image cannot behave that way either — it is downsampled from a
/// 45MP RAW, so its grain is band limited by construction.  This plane models
/// that: 18..34 strokes instead of 6..14, and one Gaussian pass over the whole
/// plane so the finest structure spans more than one pixel.
///
/// [`arb_artwork_plane`] stays as it is, because Properties 2 and 90 are about
/// determinism rather than about correlation and their fixtures are pinned.
pub fn arb_painted_artwork_plane() -> impl Strategy<Value = ArtworkPlane> {
    (any::<u64>(), 18usize..34).prop_map(|(seed, strokes)| {
        let plane = artwork_plane(seed, strokes);
        ArtworkPlane {
            image: Arc::new(image::imageops::blur(plane.image(), 0.9)),
        }
    })
}

/// One synthetic focus layer.
#[derive(Clone, Debug)]
pub struct SyntheticSource {
    /// Capture file name, unique inside one scan.
    pub filename: String,
    /// Plane this layer's pixels are cropped from, when it is not the scan's own
    /// plane.
    ///
    /// A source that depicts *other content at the same place* is how 需求 1.6's
    /// "satisfies the overlap evidence with no other source" is reached without
    /// removing the geometric overlap: the correspondences are still there and
    /// still fit, and the 需求 1.1 correlation is the gate that has to reject it.
    pub content_plane: Option<ArtworkPlane>,
    /// Ground truth tile pixel → plane pixel placement.
    pub tile_to_plane: Matrix3<f64>,
    /// Multiplicative exposure difference of this layer.
    pub exposure: f64,
    /// Gaussian defocus of this layer, in tile pixels.
    ///
    /// A Capture_Station *is* a focus bracket, so its layers do not carry the
    /// same amount of detail; 需求 2.1 picks the anchor by exactly that
    /// difference.  `0.0` is the sharp layer.
    pub defocus_sigma: f32,
    /// Capture station (grid cell) this layer belongs to.
    pub station: usize,
    /// Focus rank of this layer inside its capture station.
    pub layer: usize,
    /// Position of this layer in the serpentine capture order.
    ///
    /// The original file names are numbered along this rank, so a rename
    /// scheme can be expressed relative to it instead of relative to whatever
    /// order the sources happen to be held in.
    pub capture_rank: usize,
}

/// File naming schemes for [`SyntheticScan::with_renamed_sources`].
///
/// Requirement 1.3 asks for invariance under a *rename*, not merely under a
/// reshuffle, so the schemes below change the prefix, the numbering base and
/// the numbering direction — the three things a real "renamed the whole import"
/// does.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RenameScheme {
    /// A different camera's naming: other prefix, other number width, other
    /// extension.  The name *order* and the numeric gaps are preserved.
    ForeignPrefix,
    /// Same prefix, a numbering base far away from the original one.  The name
    /// order and the numeric gaps are preserved.
    ShiftedNumberBase,
    /// Name order preserved, but the numbering advances in steps of 10 instead
    /// of 1, so every trailing-capture-number gap lands above
    /// `FOCUS_BRACKET_MAX_CAPTURE_GAP` (6).  This is what a user who deleted
    /// the rejects before importing produces.
    WideNumberStride,
    /// Numbering that runs *against* the capture order, so the file name order
    /// is the exact reverse of the geometric scan order.  This flips
    /// `canonical_match_direction` for every pair at once.
    ReversedNumbering,
}

impl RenameScheme {
    /// Every scheme, so a property test can iterate the whole rename dimension.
    pub const ALL: [RenameScheme; 4] = [
        RenameScheme::ForeignPrefix,
        RenameScheme::ShiftedNumberBase,
        RenameScheme::WideNumberStride,
        RenameScheme::ReversedNumbering,
    ];

    fn filename(self, capture_rank: usize, capture_count: usize) -> String {
        match self {
            RenameScheme::ForeignPrefix => format!("IMG_{:05}.CR2", 1 + capture_rank),
            RenameScheme::ShiftedNumberBase => format!("DSC_{:04}.NEF", 9000 + capture_rank),
            RenameScheme::WideNumberStride => format!("DSC_{:04}.NEF", 3680 + capture_rank * 10),
            RenameScheme::ReversedNumbering => format!(
                "DSC_{:04}.NEF",
                3680 + capture_count.saturating_sub(1 + capture_rank)
            ),
        }
    }
}

/// How a source that belongs to no Capture_Station fails the 需求 1.1 evidence.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DecoySource {
    /// Overlaps one station geometrically but depicts different content, so the
    /// inlier count, the spatial support and the scale ratio all pass and the
    /// overlap correlation is the gate that rejects it.
    UnrelatedContent,
    /// Overlaps nothing, so no correspondence exists at all.
    NoOverlap,
}

impl DecoySource {
    pub const ALL: [DecoySource; 2] = [DecoySource::UnrelatedContent, DecoySource::NoOverlap];
}

/// A complete synthetic scan: the plane, its focus layers and the station pairs
/// that physically overlap.
#[derive(Clone)]
pub struct SyntheticScan {
    plane: ArtworkPlane,
    sources: Vec<SyntheticSource>,
    /// Station index pairs that are grid neighbours, ascending.
    station_neighbours: Vec<(usize, usize)>,
    /// Seed of the import order permutation this scan offers.
    import_shuffle_seed: u64,
}

impl std::fmt::Debug for SyntheticScan {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("SyntheticScan")
            .field("plane", &self.plane)
            .field("station_count", &self.station_count())
            .field("station_neighbours", &self.station_neighbours)
            .field("sources", &self.sources)
            .finish()
    }
}

impl SyntheticScan {
    pub fn plane(&self) -> &ArtworkPlane {
        &self.plane
    }

    pub fn sources(&self) -> &[SyntheticSource] {
        &self.sources
    }

    pub fn len(&self) -> usize {
        self.sources.len()
    }

    pub fn is_empty(&self) -> bool {
        self.sources.is_empty()
    }

    pub fn station_count(&self) -> usize {
        self.sources
            .iter()
            .map(|source| source.station + 1)
            .max()
            .unwrap_or(0)
    }

    /// Absolute path of one source, which is what `ImageInfo::filename` carries
    /// in production.
    pub fn source_path(&self, index: usize) -> String {
        format!("/tmp/stack-pipeline/{}", self.sources[index].filename)
    }

    /// Absolute source paths in import order, the input of
    /// `determinism::derive_run_seed_from_paths`.
    pub fn source_paths(&self) -> Vec<String> {
        (0..self.sources.len())
            .map(|index| self.source_path(index))
            .collect()
    }

    /// The same scan with the *import order* permuted and nothing else.
    ///
    /// Every source keeps its file name, its pose, its exposure and its
    /// station, so the content of the selection is bit for bit the one of
    /// `self`; only the order the pipeline receives it in differs.
    pub fn with_permuted_import_order(&self) -> Self {
        self.with_permuted_import_order_seeded(self.import_shuffle_seed)
    }

    /// [`Self::with_permuted_import_order`] with an explicit permutation seed.
    ///
    /// Requirement 1.3 asks for "at least 5 different random orders", which
    /// needs 5 seeds rather than 5 calls that all replay the one seed the scan
    /// was generated with.
    ///
    /// The permutation is guaranteed to differ from the import order whenever
    /// the scan holds at least two sources: a drawn identity permutation is
    /// replaced by the reversal.  Otherwise a lucky draw would let an
    /// invariance assertion pass without ever permuting anything.
    pub fn with_permuted_import_order_seeded(&self, seed: u64) -> Self {
        let mut order = (0..self.sources.len()).collect::<Vec<_>>();
        let mut state = seed | 1;
        // Fisher-Yates, so every permutation is reachable.
        for index in (1..order.len()).rev() {
            let swap = (splitmix64(&mut state) % (index as u64 + 1)) as usize;
            order.swap(index, swap);
        }
        if order.len() >= 2 && order.iter().enumerate().all(|(slot, &index)| slot == index) {
            order.reverse();
        }
        Self {
            plane: self.plane.clone(),
            sources: order
                .into_iter()
                .map(|index| self.sources[index].clone())
                .collect(),
            station_neighbours: self.station_neighbours.clone(),
            import_shuffle_seed: self.import_shuffle_seed,
        }
    }

    /// The same scan with every source *renamed* and nothing else.
    ///
    /// Poses, exposures, station membership, focus ranks, the plane and the
    /// import order are all preserved, so the only thing the pipeline can see
    /// differently is the file name — and everything the pipeline derives from
    /// it, including the `canonical_match_direction` fallback that feature-less
    /// synthetic tiles take.
    pub fn with_renamed_sources(&self, scheme: RenameScheme) -> Self {
        let capture_count = self.sources.len();
        Self {
            plane: self.plane.clone(),
            sources: self
                .sources
                .iter()
                .map(|source| SyntheticSource {
                    filename: scheme.filename(source.capture_rank, capture_count),
                    ..source.clone()
                })
                .collect(),
            station_neighbours: self.station_neighbours.clone(),
            import_shuffle_seed: self.import_shuffle_seed,
        }
    }

    /// File names in import order.
    pub fn filenames(&self) -> Vec<String> {
        self.sources
            .iter()
            .map(|source| source.filename.clone())
            .collect()
    }

    /// Capture identity per source, in import order.
    ///
    /// `(station, focus layer)` is what a source *is*; it survives both a
    /// rename and an import permutation, so it is the identity a membership
    /// comparison has to use.
    pub fn capture_identities(&self) -> Vec<(usize, usize)> {
        self.sources
            .iter()
            .map(|source| (source.station, source.layer))
            .collect()
    }

    /// Ground truth frame centre of one source, in plane pixels.
    ///
    /// The independent answer a property test compares the *recovered* centre
    /// order and the accumulated frame-centre displacement of 需求 1.4 against.
    pub fn plane_centre(&self, index: usize) -> Point2<f64> {
        let side = f64::from(SYNTHETIC_TILE_SIDE);
        map_point(
            &self.sources[index].tile_to_plane,
            Point2::new(side * 0.5, side * 0.5),
        )
        .expect("a synthetic pose is always invertible")
    }

    /// The same scan with one more source appended that no other source shares a
    /// Capture_Station with (需求 1.6).
    ///
    /// Returns the import index of the added source.  The rest of the selection
    /// — poses, content, names, order — is untouched, so a property test can
    /// compare the grouping of the two selections directly.
    pub fn with_isolated_source(&self, kind: DecoySource, seed: u64) -> (Self, usize) {
        let mut sources = self.sources.clone();
        let mut station_neighbours = self.station_neighbours.clone();
        let station = self.station_count();
        let side = f64::from(SYNTHETIC_TILE_SIDE);
        let capture_rank = sources.len();
        let reference = self
            .sources
            .iter()
            .position(|source| source.station == 0 && source.layer == 0)
            .unwrap_or(0);
        let (content_plane, tile_to_plane) = match kind {
            // Another subject at the same place: the correspondences exist and
            // fit, so the 需求 1.1 correlation is what has to reject the pair.
            DecoySource::UnrelatedContent => {
                station_neighbours.push((0, station));
                (
                    Some(artwork_plane(seed | 1, 9)),
                    self.sources[reference].tile_to_plane,
                )
            }
            // A frame of somewhere else entirely: no overlap with any station, so
            // there is no correspondence to measure in the first place.
            DecoySource::NoOverlap => {
                let mut pose = self.sources[reference].tile_to_plane;
                pose[(0, 2)] += side * 8.0;
                pose[(1, 2)] += side * 8.0;
                (None, pose)
            }
        };
        sources.push(SyntheticSource {
            // Sorts after every generated name, so the decoy cannot become the
            // canonical measurement direction's source or the world reference.
            filename: format!("ZZZ_{:04}.NEF", 9000 + capture_rank),
            content_plane,
            tile_to_plane,
            exposure: 1.0,
            defocus_sigma: 0.0,
            station,
            layer: 0,
            capture_rank,
        });
        (
            Self {
                plane: self.plane.clone(),
                sources,
                station_neighbours,
                import_shuffle_seed: self.import_shuffle_seed,
            },
            capture_rank,
        )
    }

    /// Reference source picked from the capture identity rather than from the
    /// file name.
    ///
    /// [`Self::reference_index`] resolves ties by file name, which is fine for
    /// a property that never renames anything but would silently move the
    /// world origin under a rename.  The first focus layer of the first station
    /// is a content identity, so it stays put under both transforms.
    pub fn geometric_reference_index(&self) -> usize {
        (0..self.sources.len())
            .min_by_key(|&index| (self.sources[index].station, self.sources[index].layer))
            .unwrap_or(0)
    }

    /// Crop one tile out of the plane through its ground truth pose.
    fn tile_image(&self, source: &SyntheticSource) -> GrayImage {
        let plane = source.content_plane.as_ref().unwrap_or(&self.plane).image();
        let tile = GrayImage::from_fn(SYNTHETIC_TILE_SIDE, SYNTHETIC_TILE_SIDE, |x, y| {
            let tile = Point2::new(f64::from(x) + 0.5, f64::from(y) + 0.5);
            let sample = map_point(&source.tile_to_plane, tile)
                .and_then(|point| registration::sample_gray(plane, point.x, point.y))
                .unwrap_or(0.0);
            Luma([(sample * source.exposure).clamp(0.0, 255.0) as u8])
        });
        if source.defocus_sigma > 0.0 {
            image::imageops::blur(&tile, source.defocus_sigma)
        } else {
            tile
        }
    }

    /// `ImageInfo` per focus layer, `id` equal to the import index.
    pub fn images(&self) -> Vec<ImageInfo> {
        self.sources
            .iter()
            .enumerate()
            .map(|(index, source)| ImageInfo {
                id: index,
                // The absolute path, exactly as a production `ImageInfo` carries
                // it: 需求 1.6 / 1.9 report by absolute path, and the tie-breaks
                // of 需求 1.2 compare the same string production compares.
                filename: self.source_path(index),
                width: SYNTHETIC_TILE_SIDE,
                height: SYNTHETIC_TILE_SIDE,
                alignment_image: self.tile_image(source),
                full_image: None,
                scale_factor: 1.0,
                focal_length_35mm: Some(75.0),
                overview_reference: false,
                features: Vec::new(),
                top_features: Vec::new(),
                foreground_range: None,
                foreground_mask: None,
                horizontal_edge_rows: Vec::new(),
                vertical_edge_columns: Vec::new(),
            })
            .collect()
    }

    /// Ground truth tile → world placement per source id, the `locked`
    /// homographies the station solver starts from.
    pub fn locked_homographies(&self) -> HashMap<usize, Matrix3<f64>> {
        self.sources
            .iter()
            .enumerate()
            .map(|(index, source)| (index, source.tile_to_plane))
            .collect()
    }

    /// The source whose file name sorts first.
    ///
    /// The reference must be picked from the *content* of the selection, never
    /// from the import order, or the whole solve would inherit that order.
    pub fn reference_index(&self) -> usize {
        (0..self.sources.len())
            .min_by(|&left, &right| {
                self.sources[left]
                    .filename
                    .cmp(&self.sources[right].filename)
            })
            .unwrap_or(0)
    }

    /// Correspondences for every physically overlapping source pair, in import
    /// order of the pair key.
    ///
    /// A pair exists when two layers share a capture station (a focus bracket
    /// link) or when their stations are grid neighbours (a station relation).
    /// Points are searched on a grid over the source tile and kept only where
    /// the mapped position lands inside the target tile — the region a real
    /// feature match could come from.
    pub fn correspondence_pairs(&self) -> Vec<((usize, usize), Vec<(Point2<f64>, Point2<f64>)>)> {
        let mut pairs = Vec::new();
        for left in 0..self.sources.len() {
            for right in (left + 1)..self.sources.len() {
                let left_station = self.sources[left].station;
                let right_station = self.sources[right].station;
                let neighbours = self.station_neighbours.contains(&(
                    left_station.min(right_station),
                    left_station.max(right_station),
                ));
                if left_station != right_station && !neighbours {
                    continue;
                }
                // A real scan does not produce a usable match between every
                // focus layer of two neighbouring stations: the sharp planes
                // differ.  Keep the layer-to-layer relations of equal focus
                // rank, which is enough for the multi-layer consensus the
                // station solver looks for and keeps the RANSAC cost bounded.
                if left_station != right_station
                    && self.sources[left].layer != self.sources[right].layer
                {
                    continue;
                }
                // Matching is asymmetric, but real decoded images choose that
                // direction from detected feature content. `capture_rank` is
                // the immutable physical identity of this synthetic capture,
                // so it models the same content-derived direction without
                // allowing a rename to change the generated correspondence
                // support itself.
                let canonical_forward =
                    self.sources[left].capture_rank <= self.sources[right].capture_rank;
                let (source, target) = if canonical_forward {
                    (left, right)
                } else {
                    (right, left)
                };
                let Some(target_inverse) = self.sources[target].tile_to_plane.try_inverse() else {
                    continue;
                };
                let source_to_target = target_inverse * self.sources[source].tile_to_plane;
                let canonical_points = overlap_correspondences(&source_to_target);
                if canonical_points.len() < 8 {
                    continue;
                }
                // Returned in the `(left, right)` key orientation; the caller
                // re-derives the canonical direction itself.
                let points = if canonical_forward {
                    canonical_points
                } else {
                    canonical_points
                        .into_iter()
                        .map(|(source_point, target_point)| (target_point, source_point))
                        .collect()
                };
                pairs.push(((left, right), points));
            }
        }
        pairs
    }
}

/// Grid search for point pairs inside the overlap of two equally sized tiles.
fn overlap_correspondences(source_to_target: &Matrix3<f64>) -> Vec<(Point2<f64>, Point2<f64>)> {
    let side = f64::from(SYNTHETIC_TILE_SIDE);
    let steps = SYNTHETIC_CORRESPONDENCE_GRID;
    let mut inside = Vec::new();
    for row in 0..steps {
        for column in 0..steps {
            let source = Point2::new(
                (f64::from(column) + 0.5) * side / f64::from(steps),
                (f64::from(row) + 0.5) * side / f64::from(steps),
            );
            let Some(target) = map_point(source_to_target, source) else {
                continue;
            };
            if target.x > 1.0 && target.y > 1.0 && target.x < side - 1.0 && target.y < side - 1.0 {
                inside.push((source, target));
            }
        }
    }
    if inside.len() <= SYNTHETIC_MAX_CORRESPONDENCES {
        return inside;
    }
    // Keep a spread subset rather than the first N, which would all sit in the
    // top rows of the overlap.
    let stride = inside.len() / SYNTHETIC_MAX_CORRESPONDENCES;
    inside
        .into_iter()
        .step_by(stride.max(1))
        .take(SYNTHETIC_MAX_CORRESPONDENCES)
        .collect()
}

/// Serpentine two dimensional scan over `plane`.
///
/// 2..3 columns × 1..2 rows of capture stations (so 2..6 stations) with
/// 18%..40% overlap, an accumulating per-station drift, a per-station exposure
/// difference and 2..3 focus layers per station.  File names follow the capture
/// order, which is exactly the order the import order permutation later
/// destroys.
pub fn arb_scan_grid(plane: ArtworkPlane) -> impl Strategy<Value = SyntheticScan> {
    (
        2usize..=3,
        1usize..=2,
        0.18f64..0.40,
        2usize..=3,
        any::<u64>(),
        any::<u64>(),
    )
        .prop_map(
            move |(columns, rows, overlap, layers, geometry_seed, shuffle_seed)| {
                build_scan_grid(
                    plane.clone(),
                    ScanGridShape {
                        columns,
                        rows,
                        overlap,
                        layers,
                        layer_jitter_px: DEFAULT_LAYER_JITTER_PX,
                        layer_defocus_step: 0.0,
                    },
                    geometry_seed,
                    shuffle_seed,
                )
            },
        )
}

/// Per-layer frame-centre jitter of [`arb_scan_grid`], in tile pixels.
///
/// Focus breathing and the rail settling between two exposures of one
/// Capture_Station; well inside `FOCUS_BRACKET_INTERNAL_MOTION_RATIO`.
const DEFAULT_LAYER_JITTER_PX: f64 = 2.0;

/// Shape of one synthetic scan: how many stations, how much they overlap and how
/// the focus layers of a station differ from each other.
#[derive(Clone, Copy, Debug)]
pub struct ScanGridShape {
    pub columns: usize,
    pub rows: usize,
    pub overlap: f64,
    pub layers: usize,
    /// Peak per-layer frame-centre jitter inside one Capture_Station, in tile
    /// pixels.  This is what 需求 1.4's accumulated displacement measures, so a
    /// generator that wants to straddle the 0.02 long-side budget varies it.
    pub layer_jitter_px: f64,
    /// Defocus added per focus rank inside one Capture_Station, in tile pixels.
    /// `0.0` makes every layer of a station equally sharp.
    pub layer_defocus_step: f32,
}

/// Serpentine scan of `shape` over `plane`, with everything derived from
/// `geometry_seed` so the result is a pure function of the drawn seed.
fn build_scan_grid(
    plane: ArtworkPlane,
    shape: ScanGridShape,
    geometry_seed: u64,
    shuffle_seed: u64,
) -> SyntheticScan {
    let ScanGridShape {
        columns,
        rows,
        overlap,
        layers,
        layer_jitter_px,
        layer_defocus_step,
    } = shape;
    let mut state = geometry_seed | 1;
    let side = f64::from(SYNTHETIC_TILE_SIDE);
    let step = side * (1.0 - overlap);
    let mut sources = Vec::new();
    let mut drift = Point2::new(0.0, 0.0);
    for row in 0..rows {
        // Serpentine: every other row is captured right to left.
        let column_order = (0..columns).collect::<Vec<_>>();
        let column_order = if row % 2 == 0 {
            column_order
        } else {
            column_order.into_iter().rev().collect()
        };
        for column in column_order {
            let station = row * columns + column;
            // Hand held drift accumulates along the scan.
            drift = Point2::new(
                drift.x + in_range(&mut state, -1.5, 1.5),
                drift.y + in_range(&mut state, -1.5, 1.5),
            );
            let origin = Point2::new(
                SYNTHETIC_SCAN_ORIGIN + column as f64 * step + drift.x,
                SYNTHETIC_SCAN_ORIGIN + row as f64 * step + drift.y,
            );
            let exposure = in_range(&mut state, 0.92, 1.08);
            for layer in 0..layers {
                // Focus breathing plus a sub-bracket handshake, both
                // well inside `FOCUS_BRACKET_INTERNAL_MOTION_RATIO`.
                let breathing = 1.0 + in_range(&mut state, -0.0015, 0.0015);
                let jitter = Point2::new(
                    in_range(&mut state, -layer_jitter_px, layer_jitter_px),
                    in_range(&mut state, -layer_jitter_px, layer_jitter_px),
                );
                // Tile → plane: scale about the tile centre, then
                // translate to the station origin.
                let centre = side * 0.5;
                let tile_to_plane = Matrix3::new(
                    breathing,
                    0.0,
                    origin.x + jitter.x + centre * (1.0 - breathing),
                    0.0,
                    breathing,
                    origin.y + jitter.y + centre * (1.0 - breathing),
                    0.0,
                    0.0,
                    1.0,
                );
                let capture_rank = sources.len();
                sources.push(SyntheticSource {
                    filename: format!("DSC_{:04}.NEF", 3680 + capture_rank),
                    content_plane: None,
                    tile_to_plane,
                    exposure: exposure * (1.0 + layer as f64 * 0.01),
                    // Focus rank 0 is the sharp layer of the station; every
                    // further rank is one step further from the focal plane.
                    defocus_sigma: layer as f32 * layer_defocus_step,
                    station,
                    layer,
                    capture_rank,
                });
            }
        }
    }
    let mut station_neighbours = Vec::new();
    for row in 0..rows {
        for column in 0..columns {
            let station = row * columns + column;
            if column + 1 < columns {
                station_neighbours.push((station, station + 1));
            }
            if row + 1 < rows {
                station_neighbours.push((station, station + columns));
            }
        }
    }
    SyntheticScan {
        plane,
        sources,
        station_neighbours,
        import_shuffle_seed: shuffle_seed,
    }
}

/// One focus bracket plus the neighbouring camera position, over `plane`.
///
/// The generator of Properties 1, 3 and 5: two Capture_Stations side by side,
/// each holding 2..6 focus layers, so one scan carries both kinds of pair the
/// membership evidence of 需求 1.1 has to separate — two layers of *one* camera
/// position, which must be accepted, and two layers of *different* camera
/// positions, which must not.
///
/// The per-layer jitter is drawn up to 3 tile pixels rather than fixed at
/// [`DEFAULT_LAYER_JITTER_PX`], so the accumulated frame-centre displacement of a
/// bracket falls on both sides of 需求 1.4's 0.02 long-side budget across a run
/// instead of clustering below it.
pub fn arb_focus_bracket(plane: ArtworkPlane) -> impl Strategy<Value = SyntheticScan> {
    (
        0.18f64..0.40,
        2usize..=6,
        0.0f64..3.0,
        0.0f32..0.45,
        any::<u64>(),
        any::<u64>(),
    )
        .prop_map(
            move |(
                overlap,
                layers,
                layer_jitter_px,
                layer_defocus_step,
                geometry_seed,
                shuffle_seed,
            )| {
                build_scan_grid(
                    plane.clone(),
                    ScanGridShape {
                        columns: 2,
                        rows: 1,
                        overlap,
                        layers,
                        layer_jitter_px,
                        layer_defocus_step,
                    },
                    geometry_seed,
                    shuffle_seed,
                )
            },
        )
}

/// A focus bracket over a freshly generated, densely painted artwork plane.
pub fn arb_artwork_focus_bracket() -> impl Strategy<Value = SyntheticScan> {
    arb_painted_artwork_plane().prop_flat_map(arb_focus_bracket)
}

// ---------------------------------------------------------------------------
// Intra-station layer pairs (需求 2.4 / 2.5 / 2.6 / 2.7)
// ---------------------------------------------------------------------------
//
// The Intra_Station_Registrar matches a *native resolution* base against one
// focus layer of the same Capture_Station, so the fixture is a pair of native
// images rather than a scan: an aperiodic paper grain and the same grain
// displaced, optionally defocused, optionally with one region that moved
// differently, and — one draw in four — replaced by unrelated content.
//
// The side is small on purpose.  One control point costs two 51x51 window
// correlations over 15x15 integer hypotheses plus the focal-plane matched
// low-pass, so the cost of a case is linear in the number of control points and
// the *number* of them is what a property about the per-control-point gates does
// not need to be large.

/// Side of an [`IntraStationLayerPair`], in native pixels.
///
/// `(20..124).step_by(NATIVE_REFINE_STEP = 16)` gives 7 grid columns of which the
/// 5 inner ones can hold a whole 67x67 patch buffer, so one pass measures 25
/// control points: enough for an 8-neighbourhood to exist in the interior and
/// cheap enough for a property test.
pub const INTRA_STATION_PAIR_SIDE: u32 = 144;

/// Largest residual displacement a bidirectional match can carry at all, in
/// patch samples.
///
/// `refine_warped_patch` reserves the whole search support before it scores a
/// hypothesis, so the reverse probe — centred on the forward match rather than on
/// the patch centre — needs
/// `INTRA_STATION_MATCH_RADIUS + INTRA_STATION_SEARCH_SAMPLES + |d| < INTRA_STATION_PATCH_HALF`.
/// With 25 + 7 < 33 that leaves strictly less than one patch sample, i.e. less
/// than `spacing` native pixels, for the displacement itself.  The generator keeps
/// the drawn displacement inside it so the fixture measures control points instead
/// of only failing to locate them.
pub const INTRA_STATION_MAX_MEASURABLE_SAMPLES: f64 = 1.0;

/// A base/layer pair of one Capture_Station.
pub struct IntraStationLayerPair {
    pub base: Rgb32FImage,
    pub layer: Rgb32FImage,
    /// Native long side of the anchor frame, which fixes 需求 2.6's error ceiling
    /// at `0.01 ×` it.
    pub anchor_long_side: u32,
    /// Displacement of the whole layer, in native pixels.
    pub shift_px: (f64, f64),
    /// Extra displacement added along `x` across the frame, in native pixels, so
    /// the displacement field is not constant and neighbouring control points
    /// disagree.
    pub gradient_shift_px: f64,
    /// Independent grain added to the layer, in normalised levels: two exposures
    /// of the same scene do not share their sensor noise, and that is what makes
    /// a bidirectional round trip miss.
    pub noise_amplitude: f32,
    /// Defocus of the layer relative to the base, in native pixels.
    pub defocus_sigma: f32,
    /// The layer depicts something else entirely.
    pub unrelated: bool,
}

impl std::fmt::Debug for IntraStationLayerPair {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("IntraStationLayerPair")
            .field("side", &self.base.width())
            .field("anchor_long_side", &self.anchor_long_side)
            .field("shift_px", &self.shift_px)
            .field("gradient_shift_px", &self.gradient_shift_px)
            .field("noise_amplitude", &self.noise_amplitude)
            .field("defocus_sigma", &self.defocus_sigma)
            .field("unrelated", &self.unrelated)
            .finish()
    }
}

/// Aperiodic paper grain.  A sum of sines would be rejected by the matcher's
/// second-peak guard for looking like a repeated stroke, which is that guard
/// doing its job rather than a registration failure.
fn grain_texture(side: u32, seed: u64) -> Rgb32FImage {
    let grain = Rgb32FImage::from_fn(side, side, |x, y| {
        let mut hash = u64::from(x).wrapping_mul(0x9E37_79B9_7F4A_7C15)
            ^ u64::from(y).wrapping_mul(0x85EB_CA6B_C2B2_AE35)
            ^ seed;
        let value = 0.25 + 0.50 * (splitmix64(&mut hash) >> 40) as f32 / 16_777_216.0;
        Rgb([value, value * 0.96, value * 0.90])
    });
    image::imageops::blur(&grain, 1.1)
}

fn sample_rgb(image: &Rgb32FImage, x: f64, y: f64) -> Rgb<f32> {
    let clamp = |value: f64, limit: u32| value.clamp(0.0, f64::from(limit) - 1.001);
    let x = clamp(x, image.width());
    let y = clamp(y, image.height());
    let x0 = x.floor() as u32;
    let y0 = y.floor() as u32;
    let fx = (x - f64::from(x0)) as f32;
    let fy = (y - f64::from(y0)) as f32;
    let at = |px: u32, py: u32| *image.get_pixel(px, py);
    let (a, b, c, d) = (
        at(x0, y0),
        at(x0 + 1, y0),
        at(x0, y0 + 1),
        at(x0 + 1, y0 + 1),
    );
    Rgb(std::array::from_fn(|channel| {
        let top = a[channel] * (1.0 - fx) + b[channel] * fx;
        let bottom = c[channel] * (1.0 - fx) + d[channel] * fx;
        top * (1.0 - fy) + bottom * fy
    }))
}

/// A base/layer pair whose displacement, displacement gradient, grain, defocus,
/// anchor long side and content relationship are all drawn.
///
/// The displacement stays below [`INTRA_STATION_MAX_MEASURABLE_SAMPLES`], because
/// the fine pass runs at one patch sample per native pixel and a larger
/// displacement cannot be measured in reverse at all — in production that pass
/// sees the residual left by the coarse one, which is exactly this size.
///
/// `anchor_long_side` is drawn from a real artwork frame's long side and from a
/// deliberately tiny one: 需求 2.6's ceiling is `0.01 ×` it, so only the tiny
/// value puts the ceiling inside the range of displacements a bidirectional match
/// can carry.  See Property 6 for what that says about the gate.
pub fn arb_intra_station_layer_pair() -> impl Strategy<Value = IntraStationLayerPair> {
    (
        any::<u64>(),
        0.0f64..0.9,
        0.0f64..0.9,
        0.0f64..0.8,
        prop::sample::select(vec![64u32, 9_504u32]),
        0.0f32..0.06,
        0.0f32..0.9,
        0u8..4,
    )
        .prop_map(
            |(
                seed,
                shift_x,
                shift_y,
                gradient_shift_px,
                anchor_long_side,
                noise_amplitude,
                defocus_sigma,
                content,
            )| {
                build_intra_station_pair(
                    seed,
                    (shift_x, shift_y),
                    gradient_shift_px,
                    anchor_long_side,
                    noise_amplitude,
                    defocus_sigma,
                    content == 0,
                )
            },
        )
}

/// One base/layer pair from an explicit set of parameters.
fn build_intra_station_pair(
    seed: u64,
    shift_px: (f64, f64),
    gradient_shift_px: f64,
    anchor_long_side: u32,
    noise_amplitude: f32,
    defocus_sigma: f32,
    unrelated: bool,
) -> IntraStationLayerPair {
    let (shift_x, shift_y) = shift_px;
    let side = INTRA_STATION_PAIR_SIDE;
    let base = grain_texture(side, seed);
    let layer = if unrelated {
        grain_texture(side, seed ^ 0xA5A5_5A5A_A5A5_5A5A)
    } else {
        Rgb32FImage::from_fn(side, side, |x, y| {
            let ramp = gradient_shift_px * f64::from(x) / f64::from(side);
            let sampled = sample_rgb(&base, f64::from(x) - shift_x - ramp, f64::from(y) - shift_y);
            if noise_amplitude <= 0.0 {
                return sampled;
            }
            let mut hash = u64::from(x).wrapping_mul(0xD6E8_FEB8_6659_FD93)
                ^ u64::from(y).wrapping_mul(0xC2B2_AE3D_27D4_EB4F)
                ^ seed.rotate_left(17);
            let noise =
                ((splitmix64(&mut hash) >> 40) as f32 / 16_777_216.0 - 0.5) * 2.0 * noise_amplitude;
            Rgb(std::array::from_fn(|channel| {
                (sampled[channel] + noise).clamp(0.0, 1.0)
            }))
        })
    };
    let layer = if defocus_sigma > 0.0 {
        image::imageops::blur(&layer, defocus_sigma)
    } else {
        layer
    };
    IntraStationLayerPair {
        base,
        layer,
        anchor_long_side,
        shift_px,
        gradient_shift_px,
        noise_amplitude,
        defocus_sigma,
        unrelated,
    }
}

/// A base/layer pair for the per-frame verdict of 需求 2.7 and 2.8.
///
/// [`arb_intra_station_layer_pair`] always displaces the layer, and a displaced
/// layer is one the local refinement can only improve: its global model still
/// carries the whole displacement, so the refined median symmetric reprojection
/// error is below the global one and 需求 2.7 keeps the field every time.  Both
/// sides of the requirement's comparison have to occur for the equivalence to be
/// asserted at all, so one draw in three makes the layer *already aligned*
/// (`shift = (0, 0)`, no gradient) and leaves only grain and defocus between the
/// two sides.  There the global model is already right, the refinement has
/// nothing left to remove, and the bidirectional residual it measures is no
/// smaller than the displacement it claims to correct — which is exactly the
/// frame 需求 2.7 reverts.
///
/// `unrelated` is drawn too, at one in six, because an unformable median is the
/// third outcome the verdict has to distinguish from a revert.
pub fn arb_intra_station_verdict_pair() -> impl Strategy<Value = IntraStationLayerPair> {
    (
        any::<u64>(),
        0.20f64..0.9,
        0.20f64..0.9,
        0.0f64..0.8,
        prop::sample::select(vec![64u32, 9_504u32]),
        0.0f32..0.06,
        0.0f32..0.9,
        0u8..6,
    )
        .prop_map(
            |(
                seed,
                shift_x,
                shift_y,
                gradient_shift_px,
                anchor_long_side,
                noise_amplitude,
                defocus_sigma,
                relationship,
            )| {
                let aligned = relationship < 2;
                build_intra_station_pair(
                    seed,
                    if aligned {
                        (0.0, 0.0)
                    } else {
                        (shift_x, shift_y)
                    },
                    if aligned { 0.0 } else { gradient_shift_px },
                    anchor_long_side,
                    noise_amplitude,
                    defocus_sigma,
                    relationship == 5,
                )
            },
        )
}

// ---------------------------------------------------------------------------
// Anchor candidates (需求 2.1)
// ---------------------------------------------------------------------------

/// Anchor candidates of one Capture_Station: 1..8 frames with distinct absolute
/// paths and quantised median Sharpness_Scores.
///
/// The scores are drawn on a 0.004 grid inside a 0.048 window, so both sides of
/// 需求 2.1's 0.01 tie window occur: several candidates inside it (the absolute
/// path decides) and a clear winner outside it (the score decides).  The path
/// order is an independent draw, so it is never the index order.
///
/// One draw in eight makes every score unmeasurable, which is the branch a
/// placeholder analysis image takes.
///
/// One draw in four instead puts a candidate *exactly* 0.01 below the best, which
/// is the only input that tells 需求 2.1's "差小于 0.01" apart from "差不超过
/// 0.01".  The scores are then 0.01 / 0.015 / 0.02, the one magnitude at which
/// `best - candidate == 0.01` holds bit for bit in binary floating point: around
/// 0.5 the nearest double to `best - 0.01` is over an ulp away from it, so no
/// draw there can ever sit on the boundary.
pub fn arb_anchor_candidates() -> impl Strategy<Value = Vec<AnchorCandidate>> {
    (
        prop::collection::vec((0usize..=12, 0u32..64), 1..=8),
        0.10f64..0.90,
        0u8..8,
        0u8..4,
    )
        .prop_map(|(draws, base, measurable, boundary)| {
            let unmeasurable = measurable == 0;
            let on_boundary = boundary == 0;
            draws
                .iter()
                .enumerate()
                .map(|(index, &(step, name))| AnchorCandidate {
                    path: format!("/tmp/stack-pipeline/DSC_{:04}_{index}.NEF", 3680 + name),
                    median_sharpness: if unmeasurable {
                        f64::NAN
                    } else if on_boundary {
                        // The first candidate holds the maximum, so a 0.01 here
                        // really is 0.01 below the best.
                        match (index, step % 3) {
                            (0, _) => 0.02,
                            (_, 0) => 0.01,
                            (_, 1) => 0.015,
                            (_, _) => 0.02,
                        }
                    } else {
                        base + step as f64 * 0.004
                    },
                })
                .collect()
        })
}

// ---------------------------------------------------------------------------
// Oversized focus bracket (需求 1.5 / 1.10)
// ---------------------------------------------------------------------------
//
// A Capture_Station holds at most 48 Source_RAWs, so the member ceiling is only
// reachable with 49 or more frames at one camera position.  Cropping 60 tiles out
// of the artwork plane and fitting 59 RANSAC models for them costs far more than
// the ceiling is worth, and none of it is under test here: the split is a
// function of the *matches*, so the fixture states them directly.
//
// Every relation of the chain is built to pass 需求 1.1 on three of its four
// measurements and to carry no measurable overlap correlation at all (a 16x16
// alignment image is below `focus_overlap_quality`'s 32px floor), which is the
// same "no correlation evidence either way" case a placeholder alignment image
// produces in production.

/// Side of the frames of an [`OversizedBracket`], in native pixels.
pub const OVERSIZED_BRACKET_SIDE: u32 = 1_000;

/// Frame-centre step of an ordinary adjacent pair of an [`OversizedBracket`], in
/// native pixels.  A multiple of 0.25 so the accumulated sums are exact in
/// binary and two equal steps compare exactly equal.
pub const OVERSIZED_BRACKET_STEP_PX: f64 = 0.25;

/// Frame-centre step of the one adjacent pair that carries the largest
/// displacement, in native pixels.
pub const OVERSIZED_BRACKET_LARGEST_STEP_PX: f64 = 1.0;

/// A single verified-link component of more than 48 members (需求 1.5 / 1.10).
pub struct OversizedBracket {
    pub images: Vec<ImageInfo>,
    pub matches: HashMap<(usize, usize), MatchInfo>,
    /// Candidate order position of the one adjacent pair carrying the largest
    /// frame-centre displacement, i.e. where 需求 1.10 must split first.
    pub largest_step_position: usize,
    /// Accumulated frame-centre displacement over the whole component, in long
    /// sides.  Below 需求 1.4's 0.02 by construction, so the *member ceiling* is
    /// the only thing that can split this component.
    pub accumulated_motion_ratio: f64,
}

impl OversizedBracket {
    pub fn len(&self) -> usize {
        self.images.len()
    }

    pub fn is_empty(&self) -> bool {
        self.images.is_empty()
    }
}

/// A summary rather than 60 frames and 59 match sets: `proptest` prints this on
/// failure.
impl std::fmt::Debug for OversizedBracket {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("OversizedBracket")
            .field("members", &self.images.len())
            .field("relations", &self.matches.len())
            .field("largest_step_position", &self.largest_step_position)
            .field("accumulated_motion_ratio", &self.accumulated_motion_ratio)
            .finish()
    }
}

fn oversized_bracket(members: usize, largest_step_position: usize) -> OversizedBracket {
    let side = OVERSIZED_BRACKET_SIDE;
    let step_of = |position: usize| {
        if position == largest_step_position {
            OVERSIZED_BRACKET_LARGEST_STEP_PX
        } else {
            OVERSIZED_BRACKET_STEP_PX
        }
    };
    // Frame centre of every member, accumulating along the candidate order.
    let mut centres = vec![0.0f64];
    for position in 1..members {
        centres.push(centres[position - 1] + step_of(position));
    }
    let images = (0..members)
        .map(|index| ImageInfo {
            id: index,
            filename: format!("/tmp/stack-pipeline/DSC_{:04}.NEF", 3680 + index),
            width: side,
            height: side,
            // Below `focus_overlap_quality`'s 32px floor on purpose: the overlap
            // correlation is then unmeasurable rather than bad.
            alignment_image: GrayImage::new(16, 16),
            full_image: None,
            scale_factor: 1.0,
            focal_length_35mm: Some(75.0),
            overview_reference: false,
            features: Vec::new(),
            top_features: Vec::new(),
            foreground_range: None,
            foreground_mask: None,
            horizontal_edge_rows: Vec::new(),
            vertical_edge_columns: Vec::new(),
        })
        .collect::<Vec<_>>();
    // A 6x6 grid over the middle 60% of the frame: 36 correspondences, so the
    // inlier count clears 30 and the inlier convex hull clears 20% of the
    // overlap.
    let grid = (0..6)
        .flat_map(|row| {
            (0..6).map(move |column| {
                Point2::new(
                    f64::from(side) * (0.2 + 0.12 * f64::from(column)),
                    f64::from(side) * (0.2 + 0.12 * f64::from(row)),
                )
            })
        })
        .collect::<Vec<_>>();
    let mut matches = HashMap::new();
    for position in 1..members {
        let left = position - 1;
        let right = position;
        // Pure translation, so the scale ratio is exactly 1.0.
        let shift = centres[left] - centres[right];
        let mut left_to_right = Matrix3::identity();
        left_to_right[(0, 2)] = shift;
        let points = grid
            .iter()
            .map(|&source| (source, Point2::new(source.x + shift, source.y)))
            .collect::<Vec<_>>();
        matches.insert(
            (left, right),
            MatchInfo {
                // The file names ascend with the index, and a synthetic frame
                // carries no features, so `canonical_match_direction` measures
                // `left -> right`: the canonical fit and the stored one agree.
                canonical_homography: Some(left_to_right),
                homography: left_to_right,
                inliers: points.len(),
                sequence_bridge: false,
                coarse_bridge: false,
                points,
                candidate_points: Vec::new(),
                top_candidate_points: Vec::new(),
                dense_focus_points: Vec::new(),
                foreground_feature_points: Vec::new(),
            },
        );
    }
    let accumulated_motion_ratio = (centres[members - 1] - centres[0]).abs() / f64::from(side);
    OversizedBracket {
        images,
        matches,
        largest_step_position,
        accumulated_motion_ratio,
    }
}

/// A verified-link component of 49..=60 members, one of whose adjacent pairs
/// carries a uniquely largest frame-centre displacement.
pub fn arb_oversized_bracket() -> impl Strategy<Value = OversizedBracket> {
    (49usize..=60)
        .prop_flat_map(|members| (Just(members), 1usize..members))
        .prop_map(|(members, largest_step_position)| {
            oversized_bracket(members, largest_step_position)
        })
}

/// A synthetic scan over a freshly generated artwork plane.
pub fn arb_artwork_scan_grid() -> impl Strategy<Value = SyntheticScan> {
    arb_artwork_plane().prop_flat_map(arb_scan_grid)
}
// ---------------------------------------------------------------------------
// Virtual_Tile generators (设计 `Testing Strategy > 生成器设计`)
// ---------------------------------------------------------------------------
//
// `arb_coverage_shape()` draws a Coverage_Mask shape, `arb_virtual_tile()`
// grows one into a complete [`VirtualTile`].  The two are separate because
// Property 13 only needs the mask geometry while Properties 18, 19 and 23 need
// the whole tile, and because the cache properties want a larger tile than the
// mask-only ones (`arb_*_in` takes the side range).
//
// Every tile a generator returns passes `VirtualTile::new`, i.e. it satisfies
// the four construction invariants.  That is deliberate: the generator produces
// *legal* tiles, and each property test derives its own illegal variants from
// them so the rejection direction is exercised too.

/// Which shape family a [`CoverageShape`] was drawn from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CoverageFamily {
    /// Nothing covered: the "no covered pixel at all" boundary of 需求 4.7.
    Empty,
    /// Every pixel covered, so the coverage bounds are the whole tile.
    Full,
    /// One axis aligned rectangle somewhere inside the tile.
    Rectangle,
    /// Two rectangles, so the union bounds are wider than either blob *and*
    /// uncovered pixels exist inside those bounds.
    TwoBlobs,
    /// Independent per pixel draw: a ragged, non convex, possibly disconnected
    /// mask.  This is the family a real projected trapezoid intersection plus a
    /// low-sharpness rejection produces.
    Ragged,
    /// Exactly one covered pixel: 1x1 bounds anywhere inside the tile.
    SinglePixel,
}

/// A Coverage_Mask shape: the pixel grid plus the covered predicate.
///
/// The covered flags are held as `bool` rather than as a [`CoverageMask`] so a
/// property test can compare the production mask against an independently
/// computed answer instead of against itself.
#[derive(Clone, PartialEq, Eq)]
pub struct CoverageShape {
    pub width: u32,
    pub height: u32,
    /// Row major, `width * height` entries.
    pub covered: Vec<bool>,
    pub family: CoverageFamily,
}

/// A summary rather than the whole bitmap: `proptest` prints this on failure.
impl std::fmt::Debug for CoverageShape {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("CoverageShape")
            .field("width", &self.width)
            .field("height", &self.height)
            .field("family", &self.family)
            .field("covered_pixels", &self.covered_count())
            .field("bounds", &self.bounds())
            .finish()
    }
}

impl CoverageShape {
    pub fn index(&self, x: u32, y: u32) -> usize {
        y as usize * self.width as usize + x as usize
    }

    pub fn is_covered(&self, x: u32, y: u32) -> bool {
        self.covered[self.index(x, y)]
    }

    pub fn covered_count(&self) -> u64 {
        self.covered.iter().filter(|&&covered| covered).count() as u64
    }

    /// Union axis aligned bounds of the covered pixels as `(x, y, w, h)`,
    /// computed here so it is an independent answer to compare
    /// `CoverageMask::covered_bounds` against (需求 4.7).
    pub fn bounds(&self) -> Option<(u32, u32, u32, u32)> {
        let mut min_x = u32::MAX;
        let mut min_y = u32::MAX;
        let mut max_x = 0u32;
        let mut max_y = 0u32;
        let mut any = false;
        for y in 0..self.height {
            for x in 0..self.width {
                if !self.is_covered(x, y) {
                    continue;
                }
                any = true;
                min_x = min_x.min(x);
                min_y = min_y.min(y);
                max_x = max_x.max(x);
                max_y = max_y.max(y);
            }
        }
        any.then(|| (min_x, min_y, max_x - min_x + 1, max_y - min_y + 1))
    }

    /// The 8 bit plane the focus renderer hands to [`CoverageMask::from_gray`].
    pub fn gray(&self) -> GrayImage {
        GrayImage::from_raw(
            self.width,
            self.height,
            self.covered
                .iter()
                .map(|&covered| if covered { 255u8 } else { 0u8 })
                .collect(),
        )
        .expect("the coverage plane has one byte per pixel")
    }

    pub fn mask(&self) -> CoverageMask {
        CoverageMask::from_gray(self.gray())
    }
}

fn filled_rectangle(shape: &mut CoverageShape, state: &mut u64) {
    let left = (splitmix64(state) % u64::from(shape.width)) as u32;
    let top = (splitmix64(state) % u64::from(shape.height)) as u32;
    let width = 1 + (splitmix64(state) % u64::from(shape.width - left)) as u32;
    let height = 1 + (splitmix64(state) % u64::from(shape.height - top)) as u32;
    for y in top..top + height {
        for x in left..left + width {
            let index = shape.index(x, y);
            shape.covered[index] = true;
        }
    }
}

fn coverage_shape(
    width: u32,
    height: u32,
    family: CoverageFamily,
    seed: u64,
    density: f64,
) -> CoverageShape {
    let mut shape = CoverageShape {
        width,
        height,
        covered: vec![false; width as usize * height as usize],
        family,
    };
    let mut state = seed | 1;
    match family {
        CoverageFamily::Empty => {}
        CoverageFamily::Full => shape.covered.fill(true),
        CoverageFamily::Rectangle => filled_rectangle(&mut shape, &mut state),
        CoverageFamily::TwoBlobs => {
            filled_rectangle(&mut shape, &mut state);
            filled_rectangle(&mut shape, &mut state);
        }
        CoverageFamily::Ragged => {
            for slot in shape.covered.iter_mut() {
                *slot = unit_interval(&mut state) < density;
            }
        }
        CoverageFamily::SinglePixel => {
            let x = (splitmix64(&mut state) % u64::from(width)) as u32;
            let y = (splitmix64(&mut state) % u64::from(height)) as u32;
            let index = shape.index(x, y);
            shape.covered[index] = true;
        }
    }
    shape
}

/// Coverage_Mask shapes over tiles of `min_side..=max_side` pixels a side.
///
/// Non square tiles are reachable (the two sides are drawn independently),
/// which is what makes a swapped row/column index in any of the four planes
/// observable.
pub fn arb_coverage_shape_in(min_side: u32, max_side: u32) -> impl Strategy<Value = CoverageShape> {
    (
        min_side..=max_side,
        min_side..=max_side,
        prop_oneof![
            1 => Just(CoverageFamily::Empty),
            1 => Just(CoverageFamily::Full),
            3 => Just(CoverageFamily::Rectangle),
            3 => Just(CoverageFamily::TwoBlobs),
            3 => Just(CoverageFamily::Ragged),
            1 => Just(CoverageFamily::SinglePixel),
        ],
        any::<u64>(),
        0.05f64..0.95,
    )
        .prop_map(|(width, height, family, seed, density)| {
            coverage_shape(width, height, family, seed, density)
        })
}

/// A fully covered shape of an explicit size, for properties that need an
/// entry of a chosen payload size rather than a drawn mask.
pub fn full_coverage_shape(width: u32, height: u32) -> CoverageShape {
    coverage_shape(width, height, CoverageFamily::Full, 1, 1.0)
}

/// Coverage_Mask shapes over 1x1..17x17 tiles.
///
/// The small side range is on purpose: the mask invariants are per pixel, and
/// 17 is not a multiple of 8, so the tail byte of the one-bit-per-pixel cache
/// encoding is exercised.  The odd sizes also reach the 1x1 and single-row
/// boundaries a large tile never visits.
pub fn arb_coverage_shape() -> impl Strategy<Value = CoverageShape> {
    arb_coverage_shape_in(1, 17)
}

/// Channel values written into covered Virtual_Tile pixels.
///
/// Each one is a value an image codec would destroy — a subnormal, a long
/// mantissa, a negative, the f32 maximum — and none of them is zero, so the
/// non-zero pixel extent of a generated tile is exactly its covered extent
/// (需求 4.7).
pub const VIRTUAL_TILE_PIXEL_PALETTE: [f32; 8] = [
    f32::from_bits(0x0000_0001),
    0.123_456_79_f32,
    -2.5_f32,
    1.0 / 3.0,
    f32::MAX,
    f32::MIN_POSITIVE,
    -1.0e-30_f32,
    9_999.999_f32,
];

/// Absolute path of legend entry `index`.
pub fn synthetic_source_path(index: usize) -> PathBuf {
    PathBuf::from(format!("/tmp/stack-pipeline/DSC_{:04}.NEF", 3680 + index))
}

fn synthetic_digest(seed: u64) -> [u8; 32] {
    let mut state = seed | 1;
    let mut digest = [0u8; 32];
    for slot in digest.iter_mut() {
        *slot = (splitmix64(&mut state) % 256) as u8;
    }
    digest
}

/// Grow a [`CoverageShape`] into a complete, invariant satisfying Virtual_Tile.
#[allow(clippy::too_many_arguments)]
fn virtual_tile_from(
    shape: &CoverageShape,
    legend_len: usize,
    station_index: usize,
    color_encoding: ColorEncoding,
    uniform_confidence: bool,
    exact_confidence: bool,
    seed: u64,
) -> VirtualTile {
    let (width, height) = (shape.width, shape.height);
    let legend = (0..legend_len)
        .map(synthetic_source_path)
        .collect::<Vec<_>>();
    let mut state = seed | 1;
    let mut owners = vec![0u16; shape.covered.len()];
    let mut pixels = Rgb32FImage::new(width, height);
    let mut confidence = vec![0.0f32; shape.covered.len()];
    for y in 0..height {
        for x in 0..width {
            let index = shape.index(x, y);
            if !shape.covered[index] {
                // Uncovered pixels stay `NO_OWNER`, confidence 0.0 and black,
                // which is what 需求 3.11 demands and what invariant 4 of
                // `VirtualTile::new` checks outside the coverage bounds.
                continue;
            }
            owners[index] = 1 + (splitmix64(&mut state) % legend_len as u64) as u16;
            let palette = splitmix64(&mut state) as usize;
            pixels.put_pixel(
                x,
                y,
                Rgb([
                    VIRTUAL_TILE_PIXEL_PALETTE[palette % VIRTUAL_TILE_PIXEL_PALETTE.len()],
                    VIRTUAL_TILE_PIXEL_PALETTE[(palette / 8) % VIRTUAL_TILE_PIXEL_PALETTE.len()],
                    VIRTUAL_TILE_PIXEL_PALETTE[(palette / 64) % VIRTUAL_TILE_PIXEL_PALETTE.len()],
                ]),
            );
            confidence[index] = if exact_confidence {
                // A multiple of 2^-10 is exactly representable as f16, so the
                // cache round trip of this value must be bit exact.
                (splitmix64(&mut state) % 1025) as f32 / 1024.0
            } else {
                // Generally *not* representable as f16: the round trip is then
                // only exact for the rounded value the cache wrote.
                (splitmix64(&mut state) % 997) as f32 / 996.0
            };
        }
    }
    let ownership = OwnershipMap::new(width, height, owners, legend.clone())
        .expect("the generated ownership map has one owner per pixel inside its legend");
    let sharpness_confidence = if uniform_confidence {
        ConfidenceMap::zero(width, height)
    } else {
        ConfidenceMap::per_pixel(width, height, confidence)
            .expect("the generated confidence map has one value per pixel")
    };
    let provenance = legend
        .iter()
        .zip(ownership.owned_pixel_counts())
        .enumerate()
        .map(|(index, (path, owned_pixels))| SourceProvenance {
            absolute_path: path.clone(),
            sha256: Some(synthetic_digest(seed ^ (index as u64 + 1))),
            owned_pixels,
        })
        .collect();
    // A translation with a long mantissa, so a lossy round trip of
    // `tile_to_world` shows up as a changed bit pattern.
    let tile_to_world = Matrix3::new(
        1.0,
        0.0,
        -1_234.567_890_123_456_7,
        0.0,
        1.0,
        -5_678.901_234_567_89,
        0.0,
        0.0,
        1.0,
    );
    VirtualTile::new(
        station_index,
        tile_to_world,
        pixels,
        ownership,
        sharpness_confidence,
        shape.mask(),
        color_encoding,
        provenance,
    )
    .expect("the generated tile satisfies every Virtual_Tile invariant")
}

/// Virtual_Tiles over tiles of `min_side..=max_side` pixels a side.
pub fn arb_virtual_tile_in(min_side: u32, max_side: u32) -> impl Strategy<Value = VirtualTile> {
    (
        arb_coverage_shape_in(min_side, max_side),
        1usize..=4,
        0usize..16,
        prop_oneof![
            Just(ColorEncoding::LinearSrgb),
            Just(ColorEncoding::DisplaySrgb)
        ],
        any::<bool>(),
        any::<bool>(),
        any::<u64>(),
    )
        .prop_map(
            |(
                shape,
                legend_len,
                station_index,
                color_encoding,
                uniform_confidence,
                exact_confidence,
                seed,
            )| {
                virtual_tile_from(
                    &shape,
                    legend_len,
                    station_index,
                    color_encoding,
                    uniform_confidence,
                    exact_confidence,
                    seed,
                )
            },
        )
}

/// Virtual_Tiles over 1x1..17x17 tiles: 1..4 Source_RAWs, both colour
/// encodings, both Sharpness_Confidence representations and every coverage
/// family.
pub fn arb_virtual_tile() -> impl Strategy<Value = VirtualTile> {
    arb_virtual_tile_in(1, 17)
}

/// A Virtual_Tile over an explicit [`CoverageShape`], for properties that draw
/// the mask themselves (Property 13).
pub fn virtual_tile_with_shape(shape: &CoverageShape, legend_len: usize) -> VirtualTile {
    virtual_tile_from(
        shape,
        legend_len,
        0,
        ColorEncoding::LinearSrgb,
        false,
        true,
        shape.covered_count().wrapping_add(u64::from(shape.width)) | 1,
    )
}

/// The smallest legal Virtual_Tile, for properties about *when* a tile is
/// resident rather than about what it holds (需求 14.4).
pub fn minimal_virtual_tile(station_index: usize) -> VirtualTile {
    let shape = coverage_shape(2, 2, CoverageFamily::SinglePixel, 1, 1.0);
    virtual_tile_from(
        &shape,
        1,
        station_index,
        ColorEncoding::LinearSrgb,
        true,
        true,
        station_index as u64 + 1,
    )
}
