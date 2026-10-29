//! Synthetic serpentine scans and focus brackets over an artwork plane.
//!
//! Every tile is cropped from the plane through a known pose, so the
//! correspondences handed to registration are geometric ground truth; renamed,
//! permuted and decoy variants are derived from a drawn scan.

use crate::panorama_utils::registration;
use crate::panorama_utils::shared::ImageInfo;
use image::GrayImage;
use image::Luma;
use nalgebra::Matrix3;
use nalgebra::Point2;
use nalgebra::Point3;
use proptest::prelude::*;
use std::collections::HashMap;
use std::sync::Arc;

use super::{ArtworkPlane, arb_artwork_plane, arb_painted_artwork_plane, in_range, splitmix64};

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

/// Plane coordinate of the top-left corner of the scanned area.
pub(super) const SYNTHETIC_SCAN_ORIGIN: f64 = 64.0;

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
pub(super) const SYNTHETIC_MAX_CORRESPONDENCES: usize = 48;

/// Grid resolution used to search for correspondences inside a tile.
pub(super) const SYNTHETIC_CORRESPONDENCE_GRID: u32 = 12;

/// Apply a homography to a point; `None` when the result is not finite.
pub(super) fn map_point(transform: &Matrix3<f64>, point: Point2<f64>) -> Option<Point2<f64>> {
    let mapped = transform * Point3::new(point.x, point.y, 1.0);
    (mapped.z.abs() > 1e-8).then(|| Point2::new(mapped.x / mapped.z, mapped.y / mapped.z))
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

    pub(super) fn filename(self, capture_rank: usize, capture_count: usize) -> String {
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
    pub(super) plane: ArtworkPlane,
    pub(super) sources: Vec<SyntheticSource>,
    /// Station index pairs that are grid neighbours, ascending.
    pub(super) station_neighbours: Vec<(usize, usize)>,
    /// Seed of the import order permutation this scan offers.
    pub(super) import_shuffle_seed: u64,
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
                // A second random artwork can share enough of the bright-paper
                // background with the primary plane to produce a spurious NCC
                // above 0.6. Build an explicitly decorrelated subject from the
                // reference content instead: polarity inversion keeps variance
                // and texture while guaranteeing that the overlap correlation
                // cannot pass merely because both fixtures are mostly paper.
                let mut state = seed | 1;
                let decorrelated =
                    GrayImage::from_fn(SYNTHETIC_TILE_SIDE, SYNTHETIC_TILE_SIDE, |x, y| {
                        let base = self.plane.image().get_pixel(x, y)[0];
                        let jitter = (splitmix64(&mut state) % 9) as i16 - 4;
                        let inverted = 255i16 - i16::from(base) + jitter;
                        Luma([inverted.clamp(0, 255) as u8])
                    });
                (
                    Some(ArtworkPlane {
                        image: Arc::new(decorrelated),
                    }),
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
    pub(super) fn tile_image(&self, source: &SyntheticSource) -> GrayImage {
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
                source_footprint: None,
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
pub(super) fn overlap_correspondences(
    source_to_target: &Matrix3<f64>,
) -> Vec<(Point2<f64>, Point2<f64>)> {
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
pub(super) const DEFAULT_LAYER_JITTER_PX: f64 = 2.0;

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
pub(super) fn build_scan_grid(
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

/// A synthetic scan over a freshly generated artwork plane.
pub fn arb_artwork_scan_grid() -> impl Strategy<Value = SyntheticScan> {
    arb_artwork_plane().prop_flat_map(arb_scan_grid)
}
