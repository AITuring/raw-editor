
/// Apply the low-frequency affine correction while preserving the owner's
/// high-frequency residual.  Pixels outside `evidence` are exact identity.
pub(crate) fn apply_low_frequency_tone(
    source: &Rgb32FImage,
    owner: &Rgb32FImage,
    evidence: &GrayImage,
    solve: &ToneSolve,
) -> Rgb32FImage {
    let width = owner.width().min(source.width()).min(evidence.width());
    let height = owner.height().min(source.height()).min(evidence.height());
    if width == 0 || height == 0 {
        return Rgb32FImage::new(width, height);
    }
    if solve.status == ToneSolveStatus::Identity {
        return image::imageops::crop_imm(owner, 0, 0, width, height).to_image();
    }
    let source_image = image::imageops::crop_imm(source, 0, 0, width, height).to_image();
    let source_low = low_frequency_field(&source_image);
    let owner_image = image::imageops::crop_imm(owner, 0, 0, width, height).to_image();
    let owner_low = low_frequency_field(&owner_image);
    Rgb32FImage::from_fn(width, height, |x, y| {
        if evidence.get_pixel(x, y)[0] == 0 {
            return *owner_image.get_pixel(x, y);
        }
        let low = source_low.get_pixel(x, y);
        let owned = owner_image.get_pixel(x, y);
        let owned_low = owner_low.get_pixel(x, y);
        Rgb(std::array::from_fn(|channel| {
            solve.gain[channel] * low[channel]
                + solve.offset[channel]
                + (owned[channel] - owned_low[channel])
        }))
    })
}

/// Explicitly named wrapper used by the compositor stage.
pub(crate) fn harmonize_owned_tone(
    source: &Rgb32FImage,
    owner: &Rgb32FImage,
    evidence: &GrayImage,
    solve: &ToneSolve,
) -> Rgb32FImage {
    apply_low_frequency_tone(source, owner, evidence, solve)
}

/// CIEDE2000 for two normalised RGB samples.  Tone boundary diagnostics use
/// the same deterministic colour difference for every owner pair; keeping it
/// here avoids a second, subtly different Delta_E implementation in a later
/// quality-gate stage.
pub(crate) fn delta_e00_rgb(left: [f32; 3], right: [f32; 3]) -> f64 {
    ciede2000(rgb_to_lab(left), rgb_to_lab(right))
}

#[derive(Clone, Copy)]
struct ToneTransform {
    /// Applied (bounded) correction `corrected = gain · value + offset`.
    gain: [f32; 3],
    offset: [f32; 3],
    /// The joint solution before the per-tile bounds (需求 9.9).
    solved_gain: [f32; 3],
    solved_offset: [f32; 3],
    /// The solved gain or offset was bounded.
    clamped: bool,
    /// The tile is in at least one relation with evidence.
    supported: bool,
}

impl ToneTransform {
    const IDENTITY: Self = Self {
        gain: [1.0; 3],
        offset: [0.0; 3],
        solved_gain: [1.0; 3],
        solved_offset: [0.0; 3],
        clamped: false,
        supported: false,
    };
}

/// Per-tile corrections from all relations at once (design: reuse the
/// `photometric.rs` joint solve).
///
/// Per channel, the log gains solve `ln gain[first] − ln gain[second] =
/// −log_gain` over every relation with Huber reweighting, each connected
/// component held at zero mean log gain, so neither station order nor an
/// anchor tile sets the level and a loop spreads its inconsistency instead of
/// passing it down a spanning tree. With the bounded gains fixed, the offsets
/// solve `offset[first] − offset[second] = median(gain[second] · second −
/// gain[first] · first)` with the same gauge. Both are then bounded per tile.
fn solve_tone_transforms(tile_count: usize, relations: &[ToneRelation]) -> Vec<ToneTransform> {
    let mut transforms = vec![ToneTransform::IDENTITY; tile_count];
    if relations.is_empty() {
        return transforms;
    }
    let mut log_gains = vec![[0.0f64; 3]; tile_count];
    for channel in 0..3 {
        let equations = relations
            .iter()
            .map(|relation| PairwiseDifference {
                first: relation.first,
                second: relation.second,
                value: -relation.log_gain[channel],
                weight: relation.weight(),
                huber_scale: Some(TONE_RELATION_HUBER_LOG + relation.scatter),
            })
            .collect::<Vec<_>>();
        let Some(solution) = solve_pairwise_differences(tile_count, &equations) else {
            return transforms;
        };
        for (tile_gains, value) in log_gains.iter_mut().zip(solution) {
            tile_gains[channel] = value;
        }
    }
    for relation in relations {
        transforms[relation.first].supported = true;
        transforms[relation.second].supported = true;
    }
    for (transform, log_gain) in transforms.iter_mut().zip(&log_gains) {
        transform.solved_gain = log_gain.map(|value| value.exp() as f32);
        transform.gain = transform
            .solved_gain
            .map(|value| value.clamp(TONE_MIN_GAIN, TONE_MAX_GAIN));
    }
    let mut offsets = vec![[0.0f64; 3]; tile_count];
    for channel in 0..3 {
        let equations = relations
            .iter()
            .map(|relation| {
                let first_gain = transforms[relation.first].gain[channel];
                let second_gain = transforms[relation.second].gain[channel];
                PairwiseDifference {
                    first: relation.first,
                    second: relation.second,
                    value: median_f64(relation.samples.iter().map(|sample| {
                        f64::from(second_gain * sample.source[channel])
                            - f64::from(first_gain * sample.owner[channel])
                    })),
                    weight: relation.weight(),
                    huber_scale: None,
                }
            })
            .collect::<Vec<_>>();
        if let Some(solution) = solve_pairwise_differences(tile_count, &equations) {
            for (tile_offsets, value) in offsets.iter_mut().zip(solution) {
                tile_offsets[channel] = value;
            }
        }
    }
    for (transform, offset) in transforms.iter_mut().zip(&offsets) {
        transform.solved_offset = offset.map(|value| value as f32);
        transform.offset = transform
            .solved_offset
            .map(|value| value.clamp(-TONE_MAX_ABS_OFFSET, TONE_MAX_ABS_OFFSET));
        transform.clamped =
            transform.gain != transform.solved_gain || transform.offset != transform.solved_offset;
    }
    transforms
}

/// Apply all owner-level tone corrections after the immutable ownership plane
/// has been selected.  Each `ToneTile` is an analysis-grid field in world
/// coordinates, so the function samples at most one low-frequency pixel per
/// output pixel and never allocates a second full-resolution RGB canvas.
/// `relations` are the accepted Station_Relations as index pairs into `tiles`.
pub(crate) fn harmonize_after_ownership(
    panorama: &mut Rgb32FImage,
    owners: &[u16],
    tiles: &[ToneTile],
    relations: &[(usize, usize)],
    world_origin: (f64, f64),
) -> ToneReport {
    harmonize_after_ownership_with_evidence(panorama, owners, tiles, relations, world_origin, None)
}

/// Production-facing entry point used by the owner compositor. `width` is
/// explicit because the ownership plane is commonly held as a flat immutable
/// slice; the height is derived from the output image. `relations` are the
/// accepted Station_Relations as index pairs into `tiles`: only their
/// overlaps are tone evidence (需求 9.1).
pub(crate) fn harmonize_tone_tiles(
    panorama: &mut Rgb32FImage,
    owners: &[u16],
    width: u32,
    tiles: &[ToneTile],
    evidence: &GrayImage,
    relations: &[(usize, usize)],
) -> ToneReport {
    if width != panorama.width() {
        let report = ToneReport {
            status: ToneStatus::Identity,
            ..ToneReport::default()
        };
        record_run_report(report.clone());
        return report;
    }
    harmonize_after_ownership_with_evidence(
        panorama,
        owners,
        tiles,
        relations,
        (0.0, 0.0),
        Some(evidence),
    )
}

