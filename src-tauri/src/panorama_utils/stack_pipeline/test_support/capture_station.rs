//! Fixtures inside one Capture_Station.
//!
//! Native base/layer pairs for the intra-station control point and verdict
//! properties (需求 2.4-2.8), anchor candidate sets (需求 2.1) and the oversized
//! verified-link bracket (需求 1.5 / 1.10).

use super::super::intra_station::AnchorCandidate;
use crate::panorama_utils::shared::ImageInfo;
use crate::panorama_utils::shared::MatchInfo;
use image::GrayImage;
use image::Rgb;
use image::Rgb32FImage;
use nalgebra::Matrix3;
use nalgebra::Point2;
use proptest::prelude::*;
use std::collections::HashMap;

use super::splitmix64;

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
pub(super) fn grain_texture(side: u32, seed: u64) -> Rgb32FImage {
    let grain = Rgb32FImage::from_fn(side, side, |x, y| {
        let mut hash = u64::from(x).wrapping_mul(0x9E37_79B9_7F4A_7C15)
            ^ u64::from(y).wrapping_mul(0x85EB_CA6B_C2B2_AE35)
            ^ seed;
        let value = 0.25 + 0.50 * (splitmix64(&mut hash) >> 40) as f32 / 16_777_216.0;
        Rgb([value, value * 0.96, value * 0.90])
    });
    image::imageops::blur(&grain, 1.1)
}

pub(super) fn sample_rgb(image: &Rgb32FImage, x: f64, y: f64) -> Rgb<f32> {
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
pub(super) fn build_intra_station_pair(
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

pub(super) fn oversized_bracket(members: usize, largest_step_position: usize) -> OversizedBracket {
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
            source_footprint: None,
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
                native_verified_points: Vec::new(),
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
