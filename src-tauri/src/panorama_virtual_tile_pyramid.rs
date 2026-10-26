//! Virtual-tile pyramid matching and relation quality.

use super::*;

/// Stable seed identifiers of a Virtual_Tile relation measurement.
pub(crate) const VIRTUAL_TILE_SEED_RENDERING_PRIOR: &str = "rendering_prior";
pub(crate) const VIRTUAL_TILE_SEED_PRIOR_FREE: &str = "prior_free_features";
/// Smallest native-pixel extent one direct-probe window must span before two
/// focus-fused stations present comparable structure (measured above; the
/// intra-station registrar needed 51px windows for the same reason, 任务 7.5).
pub(crate) const VIRTUAL_TILE_MIN_MATCH_WINDOW_NATIVE_PX: f64 = 48.0;
/// Smallest long side on which a pyramid level still carries a probe grid.
pub(crate) const VIRTUAL_TILE_PYRAMID_MIN_LEVEL_LONG_SIDE: u32 = 384;
/// Symmetric residual-model tolerance of the octaves coarser than the finest,
/// in their own pixels.
pub(crate) const VIRTUAL_TILE_PYRAMID_COARSE_INLIER_THRESHOLD_PX: f64 = 1.5;
/// Covered-feature budget and support scales of the prior-free proposal,
/// identical to the scalable source analysis path.
pub(crate) const VIRTUAL_TILE_PRIOR_FREE_MAX_FEATURES: usize = 1_600;
pub(crate) const VIRTUAL_TILE_PRIOR_FREE_SCALES: [f32; 3] =
    [0.38, 0.5, std::f32::consts::FRAC_1_SQRT_2];

/// One area-averaged level of a Virtual_Tile's covered luminance plane.
///
/// A level pixel is covered only when every native pixel of its block is, so
/// uncovered payload never reaches a probe (需求 6.1). Level pixel `p` is the
/// mean of the native block `[f·p, f·p + f)`, whose centre is
/// `f·p + (f − 1) / 2`; [`Self::level_to_native`] is exactly that map.
#[derive(Clone)]
pub(crate) struct VirtualTileMatchPlane {
    pub(crate) factor: u32,
    pub(crate) gray: GrayImage,
    pub(crate) coverage: stitching::CoverageMask,
}

impl VirtualTileMatchPlane {
    pub(crate) fn from_native(
        gray: &GrayImage,
        coverage: &stitching::CoverageMask,
        factor: u32,
    ) -> Option<Self> {
        let factor = factor.max(1);
        let (width, height) = gray.dimensions();
        if coverage.dimensions() != (width, height) {
            return None;
        }
        if factor == 1 {
            return Some(Self {
                factor,
                gray: gray.clone(),
                coverage: coverage.clone(),
            });
        }
        let level_width = width / factor;
        let level_height = height / factor;
        if level_width == 0 || level_height == 0 {
            return None;
        }
        let area = f64::from(factor * factor);
        let row_len = level_width as usize;
        let mut values = vec![0u8; row_len * level_height as usize];
        let mut covered = vec![0u8; row_len * level_height as usize];
        // Every output element is written by exactly one task and no value is
        // reduced across tasks, so the result is independent of scheduling.
        values
            .par_chunks_mut(row_len)
            .zip(covered.par_chunks_mut(row_len))
            .enumerate()
            .for_each(|(y, (value_row, covered_row))| {
                let y = y as u32;
                for x in 0..level_width {
                    let mut sum = 0u32;
                    let mut all_covered = true;
                    'block: for dy in 0..factor {
                        for dx in 0..factor {
                            let (sample_x, sample_y) = (x * factor + dx, y * factor + dy);
                            if !coverage.is_covered(sample_x, sample_y) {
                                all_covered = false;
                                break 'block;
                            }
                            sum += u32::from(gray.get_pixel(sample_x, sample_y)[0]);
                        }
                    }
                    if all_covered {
                        value_row[x as usize] = (f64::from(sum) / area).round() as u8;
                        covered_row[x as usize] = u8::MAX;
                    }
                }
            });
        Some(Self {
            factor,
            gray: GrayImage::from_raw(level_width, level_height, values)?,
            coverage: stitching::CoverageMask::from_bytes(level_width, level_height, covered)
                .ok()?,
        })
    }

    pub(crate) fn level_to_native(factor: u32) -> Matrix3<f64> {
        let scale = f64::from(factor.max(1));
        let offset = (scale - 1.0) * 0.5;
        Matrix3::new(scale, 0.0, offset, 0.0, scale, offset, 0.0, 0.0, 1.0)
    }

    pub(crate) fn native_to_level(factor: u32) -> Matrix3<f64> {
        let scale = f64::from(factor.max(1));
        let offset = (scale - 1.0) * 0.5;
        Matrix3::new(
            1.0 / scale,
            0.0,
            -offset / scale,
            0.0,
            1.0 / scale,
            -offset / scale,
            0.0,
            0.0,
            1.0,
        )
    }
}

/// Coarse-to-fine area-averaging factors, coarsest first.
///
/// The finest level makes the 13-sample probe window span at least
/// [`VIRTUAL_TILE_MIN_MATCH_WINDOW_NATIVE_PX`], but never shrinks a tile below
/// [`VIRTUAL_TILE_PYRAMID_MIN_LEVEL_LONG_SIDE`]. Coarser octaves are added while
/// they still fit: each doubles how far the ±16px search and the bounded
/// residual correction reach, so a prior that is off by a rotation or a
/// perspective remainder is corrected in steps that stay inside the bounds.
/// Small (synthetic) tiles therefore keep a single native level.
pub(crate) fn virtual_tile_pyramid_factors(shortest_long_side: u32) -> Vec<u32> {
    let window = f64::from(2 * VIRTUAL_TILE_DIRECT_PATCH_RADIUS + 1);
    let wanted = (VIRTUAL_TILE_MIN_MATCH_WINDOW_NATIVE_PX / window)
        .ceil()
        .max(1.0) as u32;
    let largest = (shortest_long_side / VIRTUAL_TILE_PYRAMID_MIN_LEVEL_LONG_SIDE).max(1);
    let mut factors = vec![wanted.min(largest).max(1)];
    while factors[0].saturating_mul(2) <= largest {
        factors.insert(0, factors[0] * 2);
    }
    factors
}

/// Probe-level metadata for one station: the bounded matcher treats a level as
/// its own pixel grid, so the plane is exposed with `scale_factor = 1`.
pub(crate) fn virtual_tile_plane_image_info(
    tile: &ImageInfo,
    plane: &VirtualTileMatchPlane,
) -> ImageInfo {
    ImageInfo {
        id: tile.id,
        filename: tile.filename.clone(),
        width: plane.gray.width(),
        height: plane.gray.height(),
        alignment_image: plane.gray.clone(),
        full_image: None,
        scale_factor: 1.0,
        focal_length_35mm: None,
        overview_reference: false,
        features: Vec::new(),
        top_features: Vec::new(),
        foreground_range: None,
        foreground_mask: None,
        horizontal_edge_rows: Vec::new(),
        vertical_edge_columns: Vec::new(),
    }
}

/// Covered luminance of a rendered Virtual_Tile; uncovered payload is zero.
pub(crate) fn virtual_tile_covered_luminance(
    pixels: &Rgb32FImage,
    coverage: &stitching::CoverageMask,
) -> GrayImage {
    let (width, height) = pixels.dimensions();
    if coverage.dimensions() != (width, height) {
        return GrayImage::new(width, height);
    }
    GrayImage::from_fn(width, height, |x, y| {
        if !coverage.is_covered(x, y) {
            return image::Luma([0]);
        }
        let pixel = pixels.get_pixel(x, y);
        let luminance = 0.2126 * pixel[0] + 0.7152 * pixel[1] + 0.0722 * pixel[2];
        image::Luma([(luminance.clamp(0.0, 1.0) * 255.0).round() as u8])
    })
}

/// FAST/BRIEF features of a covered plane, keeping only keypoints whose
/// [`VIRTUAL_TILE_FEATURE_COVERAGE_RADIUS`] window is covered, the same rule
/// as the native covered-feature path. (Scaling that window by
/// `1 / support_scale` discarded two thirds of the `langyuan-10` 1→2 inliers:
/// the overlaps are strips along tile borders.) Uncovered plane pixels are
/// zero, so a descriptor can only see a flat border, never uncovered payload.
pub(crate) fn virtual_tile_plane_features(
    plane: &VirtualTileMatchPlane,
    brief_pairs: &[(Point2<i32>, Point2<i32>)],
) -> Vec<Feature> {
    let (width, height) = plane.gray.dimensions();
    processing::find_features_multiscale(
        &plane.gray,
        brief_pairs,
        VIRTUAL_TILE_PRIOR_FREE_MAX_FEATURES,
        &VIRTUAL_TILE_PRIOR_FREE_SCALES,
    )
    .into_iter()
    .filter(|feature| {
        let radius = VIRTUAL_TILE_FEATURE_COVERAGE_RADIUS;
        let (x, y) = (feature.keypoint.x, feature.keypoint.y);
        x >= radius
            && y >= radius
            && x.saturating_add(radius) < width
            && y.saturating_add(radius) < height
            && (y - radius..=y + radius).all(|sample_y| {
                (x - radius..=x + radius)
                    .all(|sample_x| plane.coverage.is_covered(sample_x, sample_y))
            })
    })
    .collect()
}

/// Station planes of every probe level, `[level][station]`, coarsest first.
pub(crate) struct VirtualTileMatchPyramid {
    pub(crate) factors: Vec<u32>,
    pub(crate) levels: Vec<Vec<ImageInfo>>,
    pub(crate) coverages: Vec<Vec<stitching::CoverageMask>>,
    /// Native tile dimensions per station.
    pub(crate) native_dimensions: Vec<(u32, u32)>,
    /// Feature proposal grids are independent of the bounded probe grids.
    pub(crate) feature_factors: Vec<u32>,
    /// Native structural images for the final, one-native-pixel round trip.
    pub(crate) native_refinement: Vec<Option<(GrayImage, stitching::CoverageMask)>>,
    /// Coverage_Mask of each station's relation-quality plane, in the pixels
    /// of the `alignment_image` [`install_virtual_tile_match_planes`] installs.
    pub(crate) quality_coverages: Vec<stitching::CoverageMask>,
}

impl VirtualTileMatchPyramid {
    pub(crate) fn new(factors: Vec<u32>, station_count: usize) -> Self {
        let empty_coverage = || {
            stitching::CoverageMask::from_bytes(1, 1, vec![0])
                .expect("a one-pixel empty Coverage_Mask is valid")
        };
        let empty_info = || ImageInfo {
            id: usize::MAX,
            filename: String::new(),
            width: 1,
            height: 1,
            alignment_image: GrayImage::new(1, 1),
            full_image: None,
            scale_factor: 1.0,
            focal_length_35mm: None,
            overview_reference: false,
            features: Vec::new(),
            top_features: Vec::new(),
            foreground_range: None,
            foreground_mask: None,
            horizontal_edge_rows: Vec::new(),
            vertical_edge_columns: Vec::new(),
        };
        let level_count = factors.len();
        Self {
            factors,
            levels: (0..level_count)
                .map(|_| (0..station_count).map(|_| empty_info()).collect())
                .collect(),
            coverages: (0..level_count)
                .map(|_| (0..station_count).map(|_| empty_coverage()).collect())
                .collect(),
            native_dimensions: vec![(1, 1); station_count],
            feature_factors: vec![1; station_count],
            native_refinement: vec![None; station_count],
            quality_coverages: (0..station_count).map(|_| empty_coverage()).collect(),
        }
    }

    /// The masks the station solver re-checks each relation on: the
    /// relation-quality planes (requirement 6.8) and the finest match planes
    /// (requirement 6.4's overlap area).
    pub(crate) fn solver_coverages(&self) -> VirtualTileSolverCoverages<'_> {
        VirtualTileSolverCoverages {
            quality: &self.quality_coverages,
            overlap: self.coverages.last().map_or(&[][..], Vec::as_slice),
            overlap_factor: self.factors.last().copied().unwrap_or(1),
        }
    }

    /// Replace one station's planes and return them, coarsest first.
    pub(crate) fn set_station(
        &mut self,
        station: usize,
        tile: &ImageInfo,
        gray: &GrayImage,
        coverage: &stitching::CoverageMask,
    ) -> Option<Vec<VirtualTileMatchPlane>> {
        let mut planes = Vec::with_capacity(self.factors.len());
        self.native_dimensions[station] = gray.dimensions();
        self.native_refinement[station] = (gray.width().max(gray.height()) >= 4096
            && self.factors.last().is_some_and(|factor| *factor > 1))
        .then(|| {
            (
                imageproc::filter::gaussian_blur_f32(gray, 1.0),
                coverage.clone(),
            )
        });
        for (level, &factor) in self.factors.iter().enumerate() {
            let plane = VirtualTileMatchPlane::from_native(gray, coverage, factor)?;
            self.levels[level][station] = virtual_tile_plane_image_info(tile, &plane);
            self.coverages[level][station] = plane.coverage.clone();
            planes.push(plane);
        }
        (!planes.is_empty()).then_some(planes)
    }
}

/// Octaves above the finest one at which the station solver compares edges.
pub(crate) const VIRTUAL_TILE_RELATION_QUALITY_OCTAVES: u32 = 2;
/// Gaussian σ, in plane pixels, of the structural relation-quality plane.
pub(crate) const VIRTUAL_TILE_RELATION_QUALITY_BLUR_SIGMA: f32 = 1.0;
/// Half-width, in relation-quality plane pixels, of the neighbourhood that must
/// be covered around a requirement-6.8 sample: the ±1px gradient taps with
/// bilinear access read 2 pixels around it and the σ = 1 blur (imageproc
/// kernel radius ⌈2σ⌉) 2 more. On langyuan-10 the uncovered payload inside
/// that reach alone moved the verified 1→2 relation's edge-strength ratio
/// from 1.01 to 1.40 and its low-frequency difference from 0.002 to 0.13.
pub(crate) const VIRTUAL_TILE_QUALITY_COVERAGE_RADIUS_PX: f64 = 4.0;

/// The plane on which the station solver re-checks requirement 6.8.
///
/// Its `focus_overlap_quality_measurements` compares ±1-pixel gradients at a
/// sparse grid. On langyuan-10 the verified 0→2 and 1→2 relations read a
/// median edge-orientation difference of 21–38° on the unfiltered 1/4–1/16
/// planes — the level of a 40px-shifted wrong relation (38–46°), because single
/// pixel gradients follow fine silk texture. Two octaves above the finest level
/// and after a σ = 1 blur they read 4.6–5.9°, against 27–28° for the shifted
/// relation. A single matching level still needs a separate structural quality
/// plane: the probe pyramid's minimum size must not switch verification back
/// to fine silk texture. Keep at least 64 pixels on its shorter side so the
/// dense grid, gradient taps and covered blur neighbourhood remain measurable.
pub(crate) fn virtual_tile_relation_quality_plane(
    planes: &[VirtualTileMatchPlane],
    native: &GrayImage,
    coverage: &stitching::CoverageMask,
) -> (GrayImage, u32) {
    let Some(finest) = planes.last() else {
        return (GrayImage::new(1, 1), 1);
    };
    if planes.len() < 2 {
        // The probe's factor can be truncated to fit a small preview. It is
        // not a structural scale: multiplying that factor by four gave only
        // ~100 pixels across real previews and followed illumination gradients
        // instead of artwork edges. Choose the quality grid from native pixels.
        let wanted = native
            .width()
            .max(native.height())
            .div_ceil(600)
            .max(1)
            .next_power_of_two()
            .max(1 << VIRTUAL_TILE_RELATION_QUALITY_OCTAVES);
        let factor = wanted.min((native.width().min(native.height()) / 64).max(1));
        let structural = VirtualTileMatchPlane::from_native(native, coverage, factor)
            .expect("a validated probe plane has valid grayscale and coverage dimensions");
        return (
            imageproc::filter::gaussian_blur_f32(
                &structural.gray,
                VIRTUAL_TILE_RELATION_QUALITY_BLUR_SIGMA,
            ),
            factor,
        );
    }
    let wanted = finest
        .factor
        .saturating_mul(1 << VIRTUAL_TILE_RELATION_QUALITY_OCTAVES);
    let plane = planes
        .iter()
        .filter(|plane| plane.factor <= wanted)
        .max_by_key(|plane| plane.factor)
        .unwrap_or(finest);
    (
        imageproc::filter::gaussian_blur_f32(&plane.gray, VIRTUAL_TILE_RELATION_QUALITY_BLUR_SIGMA),
        plane.factor,
    )
}

/// A coarse pyramid failure may reflect the level's sparse residual model,
/// while a finer level can still measure the same prior.  Keep this retry
/// list deliberately narrow: a missing overlap or missing texture is not
/// evidence that becomes stronger at a finer level.
pub(crate) fn virtual_tile_coarse_failure_can_retry(failure_stage: &str) -> bool {
    matches!(
        failure_stage,
        // At the coarsest octave a narrow, but valid, overlap can contain
        // fewer than the final minimum number of grid probes.  That is a
        // sampling-density failure of this octave, not evidence that the
        // station pair is unrelated.  Retry at the next finer bounded octave
        // and let the unchanged final polish/Station_Relation gates decide.
        "insufficient_grid_probes"
            | "insufficient_measurable_patches"
            | "insufficient_bidirectional_matches"
            | "insufficient_grid_coverage"
            | "residual_model_no_fit"
            | "insufficient_matches_after_model_quota"
            | "insufficient_fitted_hull_support"
    )
}

/// Probe one pair coarse to fine around `seed_left_to_right` (native tile
/// pixels). Every octave but the finest runs the bounded bidirectional probe
/// (edge orientation read on the octave's σ = 1 structural plane) and re-seeds
/// the next with its measured model (tolerance
/// [`VIRTUAL_TILE_PYRAMID_COARSE_INLIER_THRESHOLD_PX`] in its own pixels); the
/// finest octave polishes that model with model-warped templates
/// ([`virtual_tile_polished_station_match`]). A single native level (small
/// tiles) keeps the bounded probe with the requirement-6.3 tolerance. The
/// relation is expressed in native tile pixels; the diagnostic describes the
/// last stage that ran.
pub(crate) fn virtual_tile_pyramid_station_match(
    left: usize,
    right: usize,
    pyramid: &VirtualTileMatchPyramid,
    seed_left_to_right: &Matrix3<f64>,
    seed_kind: &'static str,
) -> (Vec<MatchInfo>, stack_report::StationRelationCandidateRecord) {
    let empty_diagnostic = || {
        let (_, mut diagnostic) = virtual_tile_direct_station_match(
            left,
            right,
            &[],
            &[],
            &HashMap::new(),
            STATION_RELATION_MAX_MEDIAN_ERROR_PX,
        );
        diagnostic.pyramid_factors = pyramid.factors.clone();
        diagnostic.seed = seed_kind.to_string();
        diagnostic
    };
    let level_count = pyramid.factors.len();
    if level_count == 0
        || left >= pyramid.native_dimensions.len()
        || right >= pyramid.native_dimensions.len()
    {
        return (Vec::new(), empty_diagnostic());
    }
    let single_level = level_count == 1;
    let bounded_levels = if single_level { 1 } else { level_count - 1 };
    let mut seed = *seed_left_to_right;
    let mut diagnostic = empty_diagnostic();
    for level in 0..bounded_levels {
        let factor = pyramid.factors[level];
        let to_native = VirtualTileMatchPlane::level_to_native(factor);
        let from_native = VirtualTileMatchPlane::native_to_level(factor);
        let level_tiles = &pyramid.levels[level];
        let Some(level_seed_inverse) = (from_native * seed * to_native).try_inverse() else {
            diagnostic.failure_stage = "non_invertible_seed".to_string();
            return (Vec::new(), diagnostic);
        };
        let initial = HashMap::from([
            (level_tiles[left].id, Matrix3::identity()),
            (level_tiles[right].id, level_seed_inverse),
        ]);
        let inlier_threshold_px = if single_level {
            STATION_RELATION_MAX_MEDIAN_ERROR_PX
        } else {
            VIRTUAL_TILE_PYRAMID_COARSE_INLIER_THRESHOLD_PX
        };
        // A coarse octave reads edge orientation on its structural (σ = 1)
        // plane, the rule of the relation-quality plane: on langyuan-10 ±1px
        // gradients of the unfiltered 1/8 planes read 13.4–15.5° for verified
        // relations (the prior-free 1→2 proposal failed the 15° gate at 15.5°),
        // the structural planes 5.1–6.0°, and a 2-level-pixel shift 27–36°.
        let structure = (!single_level).then(|| {
            (
                imageproc::filter::gaussian_blur_f32(
                    &level_tiles[left].alignment_image,
                    VIRTUAL_TILE_RELATION_QUALITY_BLUR_SIGMA,
                ),
                imageproc::filter::gaussian_blur_f32(
                    &level_tiles[right].alignment_image,
                    VIRTUAL_TILE_RELATION_QUALITY_BLUR_SIGMA,
                ),
            )
        });
        let (relation, level_diagnostic) = virtual_tile_direct_station_match_with_structure(
            left,
            right,
            level_tiles,
            &pyramid.coverages[level],
            &initial,
            inlier_threshold_px,
            structure
                .as_ref()
                .map(|(left_plane, right_plane)| (left_plane, right_plane)),
        );
        diagnostic = level_diagnostic;
        diagnostic.pyramid_factors = pyramid.factors.clone();
        diagnostic.measurement_factor = factor;
        diagnostic.seed = seed_kind.to_string();
        let Some(relation) = relation else {
            // A later octave has a smaller residual in its own pixels and can
            // recover a model that was under-supported at this coarse level.
            // Keep the original seed until a measured relation is available;
            // no prior is ever promoted to acceptance evidence.
            if !single_level && virtual_tile_coarse_failure_can_retry(&diagnostic.failure_stage) {
                if level + 1 < bounded_levels {
                    continue;
                }
                // The last bounded octave has no finer bounded probe left.
                // Its residual-model/support failure is still a recoverable
                // measurement failure: the finest polish has independent
                // patch, reverse-consistency, hull, scale, and orientation
                // gates and must get the chance to verify the original prior.
                // Do not return the coarse diagnostic here, because that
                // would make the finest evidence unreachable for exactly the
                // sparse, shifted overlaps this retry is meant to recover.
                break;
            }
            return (Vec::new(), diagnostic);
        };
        let Some(native) = normalized_homography(&(to_native * relation.homography * from_native))
        else {
            diagnostic.failure_stage = "non_finite_native_relation".to_string();
            return (Vec::new(), diagnostic);
        };
        seed = native;
        if single_level && factor == 1 {
            let points = relation
                .points
                .iter()
                .filter_map(|(source, target)| {
                    transformed_point(&to_native, *source)
                        .zip(transformed_point(&to_native, *target))
                })
                .collect::<Vec<_>>();
            return (
                vec![MatchInfo {
                    homography: native,
                    inliers: points.len(),
                    points,
                    // Key-direction fit; see `MatchInfo::canonical_homography`.
                    canonical_homography: None,
                    ..relation
                }],
                diagnostic,
            );
        }
    }
    let finest = level_count - 1;
    let factor = pyramid.factors[finest];
    diagnostic.measurement_factor = factor;
    let relation = virtual_tile_polished_station_match(
        &pyramid.levels[finest][left],
        &pyramid.coverages[finest][left],
        &pyramid.levels[finest][right],
        &pyramid.coverages[finest][right],
        factor,
        &seed,
        (
            pyramid.native_dimensions[left],
            pyramid.native_dimensions[right],
        ),
        &mut diagnostic,
    );
    let relation = match (
        &pyramid.native_refinement[left],
        &pyramid.native_refinement[right],
    ) {
        (Some(left_native), Some(right_native)) => relation
            .into_iter()
            .filter_map(|relation| {
                virtual_tile_native_station_polish(
                    left_native,
                    right_native,
                    &relation,
                    &mut diagnostic,
                )
            })
            .collect(),
        _ => relation,
    };
    (relation, diagnostic)
}
