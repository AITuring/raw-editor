import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

// This is a cheap, dependency-free guard for the image-quality path.  The
// real-image acceptance test is intentionally opt-in (the fixtures are large),
// while these invariants catch a regression that silently turns a focus stack
// back into a coarse overwrite/average blend.
const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const read = (relativePath) => fs.readFileSync(path.join(repoRoot, relativePath), 'utf8');

const mosaic = read('src-tauri/src/panorama_utils/mosaic.rs');
const stitching = read('src-tauri/src/panorama_utils/stitching.rs');
const panorama = read('src-tauri/src/panorama_stitching.rs');
const processing = read('src-tauri/src/panorama_utils/processing.rs');
const stackPipelineDir = 'src-tauri/src/panorama_utils/stack_pipeline';
const degradation = read(`${stackPipelineDir}/degradation.rs`);
const stackReport = read(`${stackPipelineDir}/report.rs`);
const determinism = read(`${stackPipelineDir}/determinism.rs`);
const virtualTile = read(`${stackPipelineDir}/virtual_tile.rs`);
const photometric = read('src-tauri/src/panorama_utils/photometric.rs');
const tone = read(`${stackPipelineDir}/tone.rs`);
const qualityGate = read(`${stackPipelineDir}/quality_gate.rs`);

// Stage 7 Quality_Gate thresholds and pure metric dependencies are part of the
// source contract so wiring cannot silently use a relaxed acceptance band.
for (const [name, value] of [
  ['QUALITY_LOCAL_SCALE_MEDIAN_MIN', '0.98'],
  ['QUALITY_LOCAL_SCALE_PIXEL_MIN', '0.95'],
  ['QUALITY_LOCAL_SCALE_PIXEL_RATIO_MIN', '0.99'],
  ['QUALITY_EFFECTIVE_PIXEL_RATIO_MIN', '0.98'],
  ['QUALITY_MTF50_RATIO_MIN', '0.93'],
  ['QUALITY_GRADIENT_RATIO_MIN', '0.95'],
  ['QUALITY_NOISE_RATIO_MIN', '0.85'],
  ['QUALITY_NOISE_RATIO_MAX', '1.15'],
  ['QUALITY_ROI_DELTA_E00_MAX', '2.0'],
  ['QUALITY_BOUNDARY_P95_MAX', '1.5'],
  ['QUALITY_BOUNDARY_MAX', '3.0'],
  ['QUALITY_LOW_CONFIDENCE_RATIO_MAX', '0.01'],
]) {
  assert.match(qualityGate, new RegExp(`${name}[^=]*=\\s*${value}`), `${name} must remain explicit`);
}
for (const criterion of [
  'local_scale_median',
  'local_scale_pixel_ratio',
  'effective_pixel_count',
  'mtf50_normalized',
  'gradient_energy_normalized',
  'noise_sigma_ratio',
  'roi_delta_e00',
  'boundary_stroke_alignment',
  'sharpness_confidence_coverage',
]) {
  assert.ok(qualityGate.includes(`"${criterion}"`), `Quality_Gate criterion ${criterion} must remain stable`);
}
assert.match(qualityGate, /detect_lines\(/, 'MTF50 must use imageproc Hough');
assert.match(qualityGate, /acutance_with_step\(/, 'gradient energy must reuse mosaic acutance');
assert.match(qualityGate, /delta_e00_rgb\(/, 'Delta_E00 must use the canonical tone helper');

// Stage 5 photometric defaults are part of the quality contract.  Keep these
// source-level assertions dependency-free so changing an exposure bound cannot
// silently alter a real scan's calibration policy.
assert.match(photometric, /max_samples_per_pair:\s*2048/);
assert.match(photometric, /min_samples_per_pair:\s*1024/);
assert.match(photometric, /min_sample_value:\s*0\.02/);
assert.match(photometric, /max_sample_value:\s*0\.98/);
assert.match(photometric, /max_abs_log_gain:\s*1\.25_f64\.ln\(\)/);
assert.match(photometric, /allow_linear:\s*false/);
assert.doesNotMatch(tone, /std::env::var/, 'Tone_Harmonizer must not read environment switches');
for (const variable of [
  'RAW_EDITOR_ENABLE_GLOBAL_PHOTOMETRIC',
  'RAW_EDITOR_ENABLE_SPATIAL_TILE_EXPOSURE_GAIN',
  'RAW_EDITOR_SKIP_RGB_TILE_EXPOSURE_GAIN',
]) {
  assert.doesNotMatch(
    stitching,
    new RegExp(variable),
    `${variable} must not gate Tone_Harmonizer in the production stitching path`,
  );
}

// Stage 5 residual geometry keeps its activation and three node gates explicit
// so a threshold change cannot silently widen the local warp domain.
const residualWarp = read(`${stackPipelineDir}/residual_warp.rs`);
for (const [name, value] of [
  ['RESIDUAL_WARP_ENABLE_P95_PX', '3.0'],
  ['RESIDUAL_WARP_MAX_NODE_DISPLACEMENT_PX', '32.0'],
  ['RESIDUAL_WARP_MAX_NEIGHBOUR_DELTA_PX', '8.0'],
  ['RESIDUAL_WARP_MAX_ROUND_TRIP_ERROR_PX', '1.0'],
]) {
  assert.match(
    residualWarp,
    new RegExp(`(?:const|pub\\([^)]*\\) const)\\s+${name}[^=]*=\\s*${value}`),
    `${name} must remain an explicit residual-warp gate`,
  );
}
assert.match(
  residualWarp,
  /RESIDUAL_WARP_NODE_STEP_PX/,
  'Residual_Warp must use the report node spacing constant',
);
assert.match(
  residualWarp,
  /Field<2>/,
  'Residual_Warp must reuse mosaic::Field<2>',
);
const streamingGainClampCount = (mosaic.match(/STREAMING_GROUP_GAIN_MAX_LOG/g) || []).length;
assert.ok(
  streamingGainClampCount >= 4,
  'all three streaming group gain clamp sites must share ln(1.25)',
);

const numberConstant = (source, name) => {
  const value = source.match(
    new RegExp(`(?:const|pub(?:\\([^)]*\\))? const)\\s+${name}[^=]*=\\s*([0-9]+(?:\\.[0-9]+)?)`),
  )?.[1];
  assert.ok(value, `${name} must remain an explicit numeric quality guard`);
  return Number(value);
};

const selectionLongSide = numberConstant(mosaic, 'SELECTION_LONG_SIDE');
const nativeRefineStep = numberConstant(mosaic, 'NATIVE_REFINE_STEP');
const mismatchPenalty = numberConstant(mosaic, 'OWNERSHIP_MISMATCH_PENALTY');
assert.ok(selectionLongSide >= 512, 'ownership grid must remain fine enough for thin brush strokes');
assert.ok(nativeRefineStep <= 16, 'native refinement must sample at a sufficiently fine spacing');
assert.ok(mismatchPenalty >= 1, 'misregistered sharp candidates need a non-trivial mismatch penalty');

// ---------------------------------------------------------------------------
// Focus_Fuser ownership cell metrics (需求 3.1, 3.2, 3.3, 3.9; 任务 7.6, 7.7)
// ---------------------------------------------------------------------------
// The metrics live in stack_pipeline/focus_fuser.rs, not in mosaic.rs: the
// default delivery path is the layered Virtual_Tile compositor, whose per-
// station tiles always report a single capture group and therefore never reach
// mosaic::detail_preserving_mosaic. mosaic.rs keeps the primitives (and the
// legacy reductions) for the comparison path.
const focusFuser = read(`${stackPipelineDir}/focus_fuser.rs`);
const sharpnessHalfScale = numberConstant(focusFuser, 'SHARPNESS_HALF_SCALE');
const sharpnessWindowPx = numberConstant(focusFuser, 'SHARPNESS_SAMPLE_WINDOW_PX');
const ownershipCellMin = numberConstant(focusFuser, 'OWNERSHIP_CELL_MIN_PX');
const ownershipCellMax = numberConstant(focusFuser, 'OWNERSHIP_CELL_MAX_PX');
const disagreementVeto = numberConstant(focusFuser, 'OWNERSHIP_DISAGREEMENT_VETO');
assert.ok(sharpnessHalfScale > 0, 'the Sharpness_Score normalisation must stay strictly increasing');
assert.ok(sharpnessWindowPx >= 32, 'Sharpness_Score must be measured over at least a 32px window (需求 3.2)');
assert.ok(ownershipCellMin >= 8, 'ownership cells may not fall below 8 native pixels (需求 3.1)');
assert.ok(ownershipCellMax <= 64, 'ownership cells may not exceed 64 native pixels (需求 3.1)');
assert.ok(ownershipCellMin <= ownershipCellMax, 'the ownership cell band must stay non-empty');
assert.ok(disagreementVeto <= 0.2, 'the disagreement veto may not be relaxed beyond 0.2 (需求 3.3)');
// The cell size derivation has to keep both clauses of 需求 3.1 at once: a cell
// side inside [8, 64] *and* at least SELECTION_LONG_SIDE cells along the long
// side. Rounding up breaks the second clause (9504px -> 19px -> 501 cells), so
// the floor division is a source-level invariant, not a style choice.
assert.match(
  focusFuser,
  /fn ownership_cell_size_px\([\s\S]*?\.floor\(\)/,
  'the ownership cell side must round down so the long side keeps at least SELECTION_LONG_SIDE cells (需求 3.1)',
);
for (const [pattern, message] of [
  [/fn normalized_sharpness\(/, 'Sharpness_Score must be normalised to [0, 1] (需求 3.9)'],
  [/acutance \/ \(acutance \+ SHARPNESS_HALF_SCALE\)/, 'the normalisation must stay the order-preserving a / (a + k)'],
  [
    /fn sharpness_sample_step_px\(/,
    'the Sharpness_Score window must be derived from an explicit native step (需求 3.2)',
  ],
  [/fn assert_matches\(/, 'candidate and composite sampling geometry must be asserted equal (需求 3.2)'],
  [/median_of_sorted\(&differences\)/, 'cell inconsistency must be the median of the low-pass differences (需求 3.3)'],
  [/fn mismatch_penalty\(/, 'a sharper but inconsistent candidate must carry a selection penalty (需求 3.3)'],
]) {
  assert.match(focusFuser, pattern, message);
}
// The penalty condition itself: both the veto and the "candidate is sharper"
// clause must remain, otherwise a misregistered frame wins on acutance alone.
assert.match(
  focusFuser,
  /disagreement > OWNERSHIP_DISAGREEMENT_VETO && sharper/,
  'the mismatch penalty must require both a veto-level disagreement and a sharper candidate (需求 3.3)',
);
assert.match(
  focusFuser,
  /fn cell_low_pass_differences|cell_low_pass_differences\(/,
  'the Focus_Fuser must reuse the mosaic low-pass difference primitive rather than fork it',
);
assert.ok(
  !/detail_preserving_mosaic\(/.test(focusFuser),
  'the Focus_Fuser must reuse the decision primitives without the mosaic cell-wise local warp',
);

assert.match(mosaic, /fn acutance\s*\(/, 'source ownership must measure native-resolution acutance');
assert.match(mosaic, /fn ownership_disagreement\s*\(/, 'source ownership must measure pixel disagreement');
assert.match(mosaic, /seam_cut::cut_grid\(/, 'ownership decisions must be regularised by a global seam cut');
assert.match(mosaic, /mask_covered_bounds\(&mask\)/, 'in-memory mosaic output must preserve all real source coverage');
assert.match(mosaic, /store\s*\.covered_bounds\(\)/, 'streaming mosaic output must preserve all real source coverage');
assert.match(mosaic, /refine_native_layer\(/, 'the mosaic path must refine residual alignment at native resolution');
assert.match(mosaic, /Render only ownership cells/, 'final rendering must skip rejected ownership cells');
assert.match(mosaic, /every pixel has source ownership/, 'mosaic must retain an explicit full-coverage assertion');
assert.match(
  mosaic,
  /streaming_group_tone_relations_from_analysis\(/,
  'camera exposure relations must use corresponding overlap pixels before the global solve',
);
assert.match(
  mosaic,
  /same_coordinate_group_relations_recover_exposure_offset/,
  'same-coordinate group exposure estimation needs a regression test',
);

assert.match(
  stitching,
  /detail_preserving_mosaic\(/,
  'shifted focus stacks must use detail-preserving ownership rather than simple overwrite',
);
assert.match(stitching, /FOCUS_ALLOW_LOW_FREQUENCY_SEAM_BLEND/, 'seam colour correction must remain explicitly gated');

// ---------------------------------------------------------------------------
// Default Tile_Compositor path (requirements 10.2, 10.3, 11.6, 11.7, 15.1-15.3,
// 15.10, 15.11; tasks 13.1, 13.3, 13.4)
// ---------------------------------------------------------------------------
// The default path is the layered Virtual_Tile compositor, and it must deliver
// the composed canvas untouched: the union covered bounds and no second-pass
// sharpen. Both are source-level invariants because no cheap pixel test can
// tell a cropped/sharpened delivery from an honest one.
const compositorChoice = read(`${stackPipelineDir}/compositor.rs`);
assert.match(
  compositorChoice,
  /#\[default\]\s*\n\s*LayeredVirtualTile/,
  'the default compositor must stay the layered Virtual_Tile path (需求 15.2)',
);
for (const variant of ['LayeredVirtualTile', 'ProgressiveSeamTile', 'StreamingMosaic', 'LegacySingleLayerMosaic']) {
  assert.match(
    compositorChoice,
    new RegExp(`Self::${variant} => SelectedPath::${variant}`),
    `${variant} must map to its stable Stack_Report identifier (需求 15.1 / 15.3)`,
  );
}
assert.match(
  panorama,
  /StackCompositorChoice::resolve\(settings\.stack_compositor\.as_deref\(\)\)/,
  'the compositor must be chosen by a setting rather than an environment variable (需求 15.2)',
);
assert.match(
  compositorChoice,
  /Some\(0 \| 1\) => StackPathSelection \{[\s\S]*?selected_path: SelectedPath::SingleStation,[\s\S]*?use_virtual_tiles: false/,
  'a run with fewer than two Capture_Stations must select and report single_station (需求 15.11)',
);
assert.match(
  panorama,
  /select_and_record_run_path\(\s*station_count,\s*compositor_choice,\s*stack_report\.as_ref\(\),?\s*\)/,
  'production must use one boundary for compositor execution and Stack_Report path recording',
);
assert.match(
  panorama,
  /stitching::layered_virtual_tile_compositor\(/,
  'the default focus-stack path must compose through the layered Virtual_Tile compositor (任务 13.1)',
);
// 任务 13.3: the default path takes the covered union bounds and nothing else.
// `crop_to_valid_rectangle` survives for the comparison paths only.
assert.match(
  stitching,
  /TileCompositorFinishing::LayeredVirtualTile => \{\s*\n\s*crop_to_valid_bounds\(/,
  'the default path must keep the union covered bounds and must not call crop_to_valid_rectangle (需求 10.2 / 10.3)',
);
assert.match(
  stitching,
  /pub\(super\) fn crop_to_valid_rectangle\(/,
  'crop_to_valid_rectangle must be retained for the comparison paths',
);
// 任务 13.4: the default final sharpen amount is 0, so the default path never
// executes `sharpen_focus_tile_detail`. The variable stays as a diagnostic knob.
assert.match(
  stitching,
  /fn layered_final_sharpen_amount\(\) -> f32 \{[\s\S]*?RAW_EDITOR_FINAL_PANORAMA_SHARPEN_AMOUNT[\s\S]*?unwrap_or\(0\.0\)/,
  'the default final sharpen amount must be 0 (需求 11.6 / 11.7)',
);
assert.match(
  stitching,
  /fn sharpen_focus_tile_detail\(/,
  'sharpen_focus_tile_detail must be retained for the comparison paths',
);
assert.match(
  panorama,
  /report\.composition\.final_sharpen_amount = sharpen/,
  'the resolved final sharpen amount must be reported (任务 13.4)',
);
assert.match(
  stitching,
  /hard source ownership|hard ownership/i,
  'detail bands must keep hard source ownership after low-frequency blending',
);

const nativeSearchRadius = numberConstant(panorama, 'FOCUS_MATCH_REFINE_SEARCH_RADIUS');
const localResidualRatio = numberConstant(panorama, 'FOCUS_LOCAL_MATCH_RESIDUAL_RATIO');
const projectiveSupport = numberConstant(panorama, 'FOCUS_PROJECTIVE_MIN_SPATIAL_SUPPORT');
assert.ok(nativeSearchRadius >= 16, 'native focus matching must search beyond thumbnail residuals');
assert.ok(localResidualRatio > 0 && localResidualRatio <= 0.01, 'local matching gate must reject implausible warps');
assert.ok(projectiveSupport > 0 && projectiveSupport <= 0.5, 'projective alignment must require broad spatial support');
assert.match(panorama, /transform_is_stable_for_focus_stack\(/, 'focus transforms must pass a stability check');
assert.match(
  panorama,
  /order_focus_render_sources_by_capture_group\(/,
  'each camera position must finish focus selection before the next position enters the mosaic',
);
assert.match(
  panorama,
  /capture_groups_keep_long_focus_brackets/,
  'long automated focus brackets need a grouping regression',
);
assert.match(
  panorama,
  /capture_groups_split_accumulated_scan_motion/,
  'capture grouping must reject accumulated camera motion',
);
assert.match(
  panorama,
  /focus_capture_group_geometry_diagnostics\(/,
  'multi-station stacks must validate real image agreement across camera-position boundaries',
);
assert.match(
  panorama,
  /let solve_station_geometry = solve_source_station_geometry && inferred_station_count >= 2;/,
  'comparison paths may retain the source-level strict solver, but the default path must defer it',
);
assert.match(
  panorama,
  /solve_virtual_tile_station_poses_with_report\([\s\S]*?station_relations[\s\S]*?tile_homographies = solved;[\s\S]*?stitching::layered_virtual_tile_compositor\(/,
  'the default path must publish authoritative fused-tile relations and poses before composition',
);
assert.match(
  panorama,
  /enum StationRelationEvidenceKind \{[\s\S]*?VirtualTile,[\s\S]*?fn minimum_independent_support\(self\)[\s\S]*?Self::VirtualTile => 1/,
  'Virtual_Tile evidence must be explicit and must not fabricate Source_RAW consensus support',
);
assert.match(
  panorama,
  /default_two_stage_flow_solves_ten_sources_as_three_fused_stations/,
  'a compact 10-frame / 3-station run needs a two-stage fused-evidence regression',
);
assert.match(
  panorama,
  /single_virtual_tile_skips_station_relations_and_closure/,
  'a single Capture_Station must keep the inter-station solver and closure path idle',
);
assert.match(
  panorama,
  /focus_station_geometry_guard_does_not_hide_broken_boundaries/,
  'same-position focus edges must not hide broken camera-position boundaries',
);
assert.match(
  panorama,
  /focus_stack_geometry_uses_overlap_evidence_and_rendering_uses_capture_order/,
  'stack geometry must come from image overlap evidence',
);
assert.match(
  panorama,
  /focus_alignment_does_not_trade_fit_precision_for_a_few_more_inliers/,
  'fit selection must protect precision over raw inlier count',
);
assert.match(
  panorama,
  /focus_stack_stability_rejects_extrapolated_projective_warp/,
  'extrapolated projective warps need a regression test',
);

// ---------------------------------------------------------------------------
// Capture_Station membership evidence (requirement 1.1, 1.2, 1.4, 1.5, 1.10)
// ---------------------------------------------------------------------------
// Requirement 1.1 lists the *only* four pieces of image evidence a station
// membership decision may rest on. Pin every threshold at source level: a
// relaxation here silently lets two camera positions share ownership cells,
// which no pixel test downstream can distinguish from a focus bracket.
const stationMinInliers = numberConstant(panorama, 'STATION_MIN_INLIER_MATCHES');
const stationMinOverlapNcc = numberConstant(panorama, 'STATION_MIN_OVERLAP_NCC');
const stationMinSpatialSupport = numberConstant(panorama, 'STATION_MIN_INLIER_SPATIAL_SUPPORT');
const stationScaleRatioMin = numberConstant(panorama, 'STATION_SCALE_RATIO_MIN');
const stationScaleRatioMax = numberConstant(panorama, 'STATION_SCALE_RATIO_MAX');
const stationAccumulatedMotion = numberConstant(panorama, 'STATION_MAX_ACCUMULATED_CENTER_MOTION_RATIO');
const stationScoreTieBreak = numberConstant(panorama, 'STATION_EVIDENCE_SCORE_TIE_BREAK');
const stationMaxMembers = numberConstant(panorama, 'FOCUS_BRACKET_MAX_COMPONENT_SOURCES');
assert.ok(stationMinInliers >= 30, 'station membership needs at least 30 verified inlier pairs');
assert.ok(stationMinOverlapNcc >= 0.6, 'station membership needs an overlap correlation of at least 0.60');
assert.ok(stationMinSpatialSupport >= 0.2, 'station membership needs at least 20% inlier spatial support');
assert.ok(stationScaleRatioMin >= 0.98, 'the intra-station scale window must not open below 0.98');
assert.ok(stationScaleRatioMax <= 1.02, 'the intra-station scale window must not open above 1.02');
assert.ok(stationAccumulatedMotion <= 0.02, 'accumulated centre motion must split a station at 0.02 long sides');
assert.ok(stationScoreTieBreak <= 0.001, 'an absolute-path tie-break is only allowed within 0.001 of score');
assert.ok(stationMaxMembers <= 48, 'a Capture_Station must hold at most 48 sources');
assert.match(
  panorama,
  /fn homography_scale_ratio\(/,
  'the scale-ratio criterion must be one reusable measurement shared by the intra- and inter-station thresholds',
);
assert.match(
  panorama,
  /fn focus_station_link_evidence\(/,
  'station membership must expose its four measurements rather than only a boolean',
);
assert.match(
  panorama,
  /capture_station_link_requires_all_four_pieces_of_evidence/,
  'each of the four station evidence gates needs a regression test',
);

// Station_Relation production acceptance (requirements 6.3-6.5, 6.8-6.10).
const stationRelationMinInliers = numberConstant(panorama, 'STATION_RELATION_MIN_INLIERS');
const stationRelationMaxMedianError = numberConstant(panorama, 'STATION_RELATION_MAX_MEDIAN_ERROR_PX');
const stationRelationScaleMin = numberConstant(panorama, 'STATION_RELATION_SCALE_RATIO_MIN');
const stationRelationScaleMax = numberConstant(panorama, 'STATION_RELATION_SCALE_RATIO_MAX');
const stationRelationMinSupport = numberConstant(panorama, 'STATION_RELATION_MIN_SPATIAL_SUPPORT');
const stationRelationMaxMeanDifference = numberConstant(
  panorama,
  'STATION_RELATION_MAX_LOW_FREQUENCY_MEAN_RELATIVE_DIFFERENCE',
);
const stationRelationEdgeStrengthMin = numberConstant(panorama, 'STATION_RELATION_EDGE_STRENGTH_RATIO_MIN');
const stationRelationEdgeStrengthMax = numberConstant(panorama, 'STATION_RELATION_EDGE_STRENGTH_RATIO_MAX');
const stationRelationMaxOrientationDifference = numberConstant(
  panorama,
  'STATION_RELATION_MAX_MEDIAN_EDGE_ORIENTATION_DIFFERENCE_DEGREES',
);
assert.equal(stationRelationMinInliers, 24, 'a Station_Relation requires at least 24 inliers');
assert.equal(stationRelationMaxMedianError, 3.0, 'relation error is an absolute 3px world-space gate');
assert.equal(stationRelationScaleMin, 0.95, 'relation scale must not fall below 0.95');
assert.equal(stationRelationScaleMax, 1.05, 'relation scale must not exceed 1.05');
assert.equal(stationRelationMinSupport, 0.2, 'relation inlier hull must cover 20% of overlap');
assert.equal(stationRelationMaxMeanDifference, 0.2, 'low-frequency mean difference is capped at 20%');
assert.equal(stationRelationEdgeStrengthMin, 0.7, 'edge-strength ratio must not fall below 0.7');
assert.equal(stationRelationEdgeStrengthMax, 1.4, 'edge-strength ratio must not exceed 1.4');
assert.equal(
  stationRelationMaxOrientationDifference,
  10.0,
  'median edge orientation difference is capped at 10 degrees',
);

// Closure_Optimizer correction and convergence guards (requirements 7.4, 7.5).
const closureResidualMedianLimit = numberConstant(panorama, 'CLOSURE_RESIDUAL_MEDIAN_LIMIT_PX');
const closureMaxCornerCorrection = numberConstant(panorama, 'CLOSURE_MAX_CORNER_CORRECTION_PX');
const closureRelativeResidualDecrease = numberConstant(panorama, 'CLOSURE_RELATIVE_RESIDUAL_DECREASE');
const closureMaxIterations = numberConstant(panorama, 'CLOSURE_MAX_ITERATIONS');
assert.equal(closureResidualMedianLimit, 2.0, 'closure residual acceptance stays expressed as 2 world pixels');
assert.equal(closureMaxCornerCorrection, 256, 'station corner corrections may never exceed 256 world pixels');
assert.equal(closureRelativeResidualDecrease, 0.0001, 'closure convergence requires less than 1e-4 relative decrease');
assert.equal(closureMaxIterations, 100, 'closure optimization must stop after at most 100 iterations');
assert.doesNotMatch(
  panorama,
  /FOCUS_GROUP_MAX_CENTER_CORRECTION_RATIO/,
  'the old center-based correction ratio must not remain in the station closure',
);
assert.match(
  panorama,
  /fn maximum_station_corner_displacement\([\s\S]*?Point2::new\(0\.0, 0\.0\)[\s\S]*?Point2::new\(dimensions\.0 as f64, dimensions\.1 as f64\)/,
  'station correction limits must measure all four projected corners (需求 7.4)',
);
assert.match(
  panorama,
  /closure_report\.iterations = iterations[\s\S]*?closure_report\.residual_median_px = median_residual[\s\S]*?closure_report\.residual_p95_px = residual_p95/,
  'closure convergence and residual diagnostics must reach Stack_Report (需求 7.5)',
);
assert.match(
  panorama,
  /degradation::CLOSURE_CORRECTION_CLAMPED/,
  'a clamped station correction must emit closure_correction_clamped',
);
assert.match(
  panorama,
  /fn solve_robust_closure_translations\([\s\S]*?closure_residual_weight\(magnitude\)[\s\S]*?CLOSURE_MAX_ITERATIONS/,
  'the closure solve must integrate the robust weight and bounded convergence loop',
);
assert.doesNotMatch(
  panorama,
  /if dx\.hypot\(dy\) > maximum_constraint/,
  'accepted Station_Relations must not be discarded before robust closure weighting (需求 7.2)',
);
assert.doesNotMatch(
  panorama,
  /for _ in 0\.\.8[\s\S]{0,3000}maximum_change < 0\.01/,
  'the station closure must not regress to fixed iterations and an absolute movement threshold',
);
assert.doesNotMatch(
  panorama,
  /FOCUS_GROUP_CONSENSUS_MAX_ERROR_RATIO/,
  'Station_Relation acceptance must not scale its pixel error gate by image dimensions',
);
assert.match(
  panorama,
  /fn station_relation_rejection_reasons\(/,
  'each failed relation defect must map to a stable station_relation identifier',
);
assert.match(
  panorama,
  /rejected_by_reason[\s\S]*\.entry\(reason\.to_string\(\)\)/,
  'rejected relation defects must be aggregated by stable reason',
);
assert.match(
  panorama,
  /capture_sequence_weight_boost_never_reaches_station_membership/,
  'the capture-sequence weight boost must stay out of station membership',
);
assert.match(
  panorama,
  /accumulated_motion_split_records_its_position_and_ratio/,
  'the accumulated-motion split position must be pinned by a regression test',
);
assert.match(
  panorama,
  /oversized_station_is_split_until_every_member_count_fits/,
  'the 48-member ceiling needs a regression test',
);
assert.match(
  panorama,
  /grouping_reports_an_isolated_source_without_moving_the_others/,
  'an isolated source must be reported without disturbing the rest of the grouping',
);

// ---------------------------------------------------------------------------
// Failure reason identifiers (requirement 12.7, asserted via 15.9)
// ---------------------------------------------------------------------------
// The identifiers are the machine readable contract between the pipeline and
// every consumer of a Stack_Report, so their shape and uniqueness are checked at
// the source level: no RAW decode and no stitching run needed.

const identifierDeclarations = [...degradation.matchAll(/pub const ([A-Z][A-Z0-9_]*): &str =\s*"([^"]*)"/g)].map(
  ([, name, value]) => ({ name, value }),
);

assert.ok(
  identifierDeclarations.length >= 50,
  `the failure identifier set must stay complete, found ${identifierDeclarations.length}`,
);

const identifierPattern = /^[a-z0-9_]{1,64}$/;
for (const { name, value } of identifierDeclarations) {
  assert.match(value, identifierPattern, `${name} = "${value}" must match ^[a-z0-9_]{1,64}$ to stay machine readable`);
  assert.equal(
    value,
    name.toLowerCase(),
    `${name} must carry its own name as the identifier so the two can never drift apart`,
  );
}

const duplicateIdentifiers = identifierDeclarations
  .map(({ value }) => value)
  .filter((value, index, all) => all.indexOf(value) !== index);
assert.deepEqual(
  duplicateIdentifiers,
  [],
  `failure reason identifiers must be unique: ${duplicateIdentifiers.join(', ')}`,
);

const reasonTable = (name) => {
  const body = degradation.match(new RegExp(`${name}:[^=]*=\\s*&\\[([\\s\\S]*?)\\n\\];`))?.[1];
  assert.ok(body, `${name} must remain an explicit table of identifiers`);
  return [...body.matchAll(/\b([A-Z][A-Z0-9_]*)\b/g)].map(([, entry]) => entry);
};

const tabulatedReasonNames = new Set([...reasonTable('FAILURE_REASONS'), ...reasonTable('UNMEASURABLE_REASONS')]);
for (const { name } of identifierDeclarations) {
  assert.ok(
    tabulatedReasonNames.has(name),
    `${name} must appear in FAILURE_REASONS or UNMEASURABLE_REASONS, otherwise its severity is undefined`,
  );
}

assert.match(
  degradation,
  /fn is_valid_reason_identifier\(/,
  'the identifier format must also be enforceable at runtime, not only by this test',
);
assert.match(
  degradation,
  /MAX_REASON_IDENTIFIER_LENGTH: usize = 64/,
  'the 64-byte identifier ceiling of requirement 12.7 must stay an explicit constant',
);

// Severity vocabulary: `informational` separates housekeeping from a genuine
// loss of output quality, so a run that only made room in the cache still
// reports `success`.
for (const identifier of ['informational', 'degraded', 'rejected']) {
  assert.ok(
    degradation.includes(`"${identifier}"`),
    `the severity vocabulary must keep the stable identifier "${identifier}"`,
  );
}

const tabulatedSeverities = [...degradation.matchAll(/\(\s*([A-Z][A-Z0-9_]*)\s*,\s*Severity::(\w+)\s*,?\s*\)/g)].map(
  ([, name, severity]) => ({ name, severity }),
);
assert.deepEqual(
  tabulatedSeverities.filter(({ severity }) => severity === 'Informational').map(({ name }) => name),
  ['CACHE_EVICTED_FOR_CAPACITY'],
  'only cache capacity eviction is pure housekeeping; every other reason costs the output something',
);
assert.match(
  degradation,
  /fn has_degradation\(&self\)[\s\S]*?Severity::Degraded/,
  'the ledger must ask for genuine degradations rather than for a non-empty entry list',
);
const ledgerResult = degradation.match(/pub fn result\(&self\) -> RunResult \{([\s\S]*?)\n    \}/)?.[1];
assert.ok(ledgerResult, 'degradation.rs must keep DegradationLedger::result as the single verdict rule');
assert.doesNotMatch(
  ledgerResult,
  /entries\.is_empty\(\)/,
  'the verdict must not treat an informational-only ledger as a degraded run',
);
assert.match(
  ledgerResult,
  /has_degradation\(\)/,
  'the verdict must be decided by degraded entries, not by entry count',
);

// ---------------------------------------------------------------------------
// Stack_Report top-level schema (requirements 10.8 / 11.16, asserted via 15.9)
// ---------------------------------------------------------------------------

const expectedTopLevelFields = [
  'schema',
  'pipeline_version',
  'run_id',
  'started_at_epoch_ms',
  'finished_at_epoch_ms',
  'selected_path',
  'input',
  'grouping',
  'intra_station',
  'fusion',
  'virtual_tiles',
  'virtual_tile_cache',
  'topology',
  'station_relations',
  'closure',
  'residual_warp',
  'tone',
  'composition',
  'quality_gate',
  'output',
  'resources',
  'degradation',
];

const topLevelFieldList = stackReport.match(/STACK_REPORT_TOP_LEVEL_FIELDS:[^=]*=\s*&\[([\s\S]*?)\n\];/)?.[1];
assert.ok(topLevelFieldList, 'report.rs must keep STACK_REPORT_TOP_LEVEL_FIELDS as an explicit schema list');
const declaredTopLevelFields = [...topLevelFieldList.matchAll(/"([a-z0-9_]+)"/g)].map(([, field]) => field);
assert.deepEqual(
  declaredTopLevelFields,
  expectedTopLevelFields,
  'the Stack_Report top-level schema must match the design document exactly, in order',
);

const stackReportStructBody = stackReport.match(/pub\(crate\) struct StackReport \{([\s\S]*?)\n\}/)?.[1];
assert.ok(stackReportStructBody, 'report.rs must declare the StackReport struct');
const stackReportStructFields = [...stackReportStructBody.matchAll(/^\s*pub ([a-z0-9_]+):/gm)].map(
  ([, field]) => field,
);
assert.deepEqual(
  stackReportStructFields,
  expectedTopLevelFields,
  'every schema field must exist on StackReport so a default report always serialises it',
);
assert.doesNotMatch(
  stackReportStructBody,
  /skip_serializing/,
  'no top-level field may be dropped from the serialised report, defaults included',
);

for (const identifier of ['running', 'success', 'degraded', 'rejected', 'cancelled']) {
  assert.ok(
    stackReport.includes(`"${identifier}"`),
    `the run result vocabulary must keep the stable identifier "${identifier}"`,
  );
}
assert.match(
  stackReport,
  /pub entries: Vec<DegradationEntryRecord>/,
  'the report must carry the collected degradation entries, not only a verdict',
);
assert.match(
  stackReport,
  /fn apply_degradation_ledger\(/,
  'the DegradationLedger must be bridged into the report on every terminating path',
);

// ---------------------------------------------------------------------------
// Determinism baseline (requirement 14.6, asserted via 15.9)
// ---------------------------------------------------------------------------
// Three runtime quantities can silently reorder the pipeline and change the
// output bits: the per-process `HashMap` hash seed, the `rayon` thread count,
// and the RANSAC sampling seed.  None of them is visible in a single run, so a
// regression here cannot be caught by comparing one output against a reference
// image — it needs a source level guard.  Everything below reads Rust sources
// only: no RAW decode, no network, no stitching run.

// Blank out comments so that a banned symbol *named in prose* can neither
// satisfy nor break a code level assertion.  The doc comment of
// `DETERMINISTIC_SUM_BLOCK_LEN`, for example, explains that the constant may
// not come from `rayon::current_num_threads()` — the very identifier the
// assertion below forbids in code.
//
// String literals are preserved (the identifier tables are asserted elsewhere)
// and `\\`-escapes inside them are skipped, so a `"` inside a literal cannot
// desynchronise the scan.  These sources contain no raw strings and no char
// literal holding a quote or a slash, which is all this simplification needs.
const stripRustComments = (source) => {
  let output = '';
  let cursor = 0;
  let inString = false;
  const blank = (character) => (character === '\n' ? '\n' : ' ');
  while (cursor < source.length) {
    const pair = source.slice(cursor, cursor + 2);
    if (inString) {
      if (source[cursor] === '\\') {
        output += source.slice(cursor, cursor + 2);
        cursor += 2;
        continue;
      }
      if (source[cursor] === '"') inString = false;
      output += source[cursor];
      cursor += 1;
      continue;
    }
    if (pair === '//') {
      while (cursor < source.length && source[cursor] !== '\n') {
        output += ' ';
        cursor += 1;
      }
      continue;
    }
    if (pair === '/*') {
      let depth = 1;
      output += '  ';
      cursor += 2;
      while (cursor < source.length && depth > 0) {
        const inner = source.slice(cursor, cursor + 2);
        if (inner === '/*' || inner === '*/') {
          depth += inner === '/*' ? 1 : -1;
          output += '  ';
          cursor += 2;
          continue;
        }
        output += blank(source[cursor]);
        cursor += 1;
      }
      continue;
    }
    if (source[cursor] === '"') inString = true;
    output += source[cursor];
    cursor += 1;
  }
  return output;
};

// Everything below the `#[cfg(test)]` module is test scaffolding, which is
// allowed to do what the shipped code may not: the determinism unit tests build
// fixed size `rayon` pools (`num_threads(..)`) precisely to prove the result
// does not move with the thread count.
const withoutTestModule = (code) => code.split(/^#\[cfg\(test\)\]/m)[0];

const determinismCode = withoutTestModule(stripRustComments(determinism));
const processingCode = stripRustComments(processing);
const panoramaCode = stripRustComments(panorama);

// --- 1. no direct `for ... in` iteration over a hash container -------------
// `HashMap`/`HashSet` iteration order is a function of the process wide hash
// seed, so any decision or floating point reduction taken in iteration order is
// irreproducible across runs of the same input.  Inside `stack_pipeline` the
// only sanctioned ways to walk a hash container are the `determinism` helpers
// (which sort first) or switching the container to `BTreeMap`/`BTreeSet`.

const APPROVED_ORDERING_HELPERS = /\b(?:sorted_pairs|sorted_keys|into_sorted_pairs)\s*\(/;

// Names bound to a hash container: `let x = HashMap::new()`, `let x: HashSet<_>
// = ...`, function parameters and struct fields typed `HashMap<..>`.
const hashContainerBindings = (code) => {
  const names = new Set();
  for (const [, name] of code.matchAll(
    /\blet\s+(?:mut\s+)?([a-z_][a-z0-9_]*)\s*(?::[^=;]*)?=\s*[^;]{0,200}?\bHash(?:Map|Set)::/g,
  )) {
    names.add(name);
  }
  for (const [, name] of code.matchAll(/\b([a-z_][a-z0-9_]*)\s*:\s*&?(?:mut\s+)?Hash(?:Map|Set)\s*</g)) {
    names.add(name);
  }
  return names;
};

const stackPipelineFiles = fs
  .readdirSync(path.join(repoRoot, stackPipelineDir))
  .filter((entry) => entry.endsWith('.rs'))
  .sort();
assert.ok(stackPipelineFiles.length >= 6, 'the stack_pipeline module must keep its files discoverable by this guard');

// Standalone modules declared behind `#[cfg(test)]` are not shipped code. The
// HashMap-order contract must ignore them just as it ignores inline test
// modules, while continuing to scan every production module discovered above.
const stackPipelineModule = read(`${stackPipelineDir}/mod.rs`);
const testOnlyStackPipelineFiles = new Set(
  [...stackPipelineModule.matchAll(/#\[cfg\(test\)\]\s*(?:pub\(crate\)\s+)?mod\s+([a-z_][a-z0-9_]*)\s*;/g)].map(
    ([, moduleName]) => `${moduleName}.rs`,
  ),
);
assert.deepEqual(
  [...testOnlyStackPipelineFiles].sort(),
  ['properties.rs', 'test_support.rs'],
  'the production hash-order guard must derive the standalone test-only modules from mod.rs',
);
const productionStackPipelineFiles = stackPipelineFiles.filter((file) => !testOnlyStackPipelineFiles.has(file));

const hashIterationViolations = [];
let inspectedHashBindings = 0;
for (const file of productionStackPipelineFiles) {
  const code = withoutTestModule(stripRustComments(read(`${stackPipelineDir}/${file}`)));
  const bindings = hashContainerBindings(code);
  inspectedHashBindings += bindings.size;
  if (bindings.size === 0) continue;
  code.split('\n').forEach((line, index) => {
    const iterated = line.match(/\bfor\s+.*?\s+in\s+(.*)$/)?.[1];
    if (!iterated || APPROVED_ORDERING_HELPERS.test(iterated)) return;
    for (const name of bindings) {
      // `name` on its own, or as the receiver of `.iter()` / `.keys()` /
      // `.values()` / `.drain()` — all of which inherit the hash order.
      if (new RegExp(`(?:^|[^a-z0-9_.])${name}\\b`).test(iterated)) {
        hashIterationViolations.push(`${file}:${index + 1}: for ... in ${iterated.trim()}`);
        break;
      }
    }
  });
}
assert.deepEqual(
  hashIterationViolations,
  [],
  'stack_pipeline must not iterate a HashMap/HashSet directly; use determinism::sorted_pairs / ' +
    `sorted_keys / into_sorted_pairs or a BTreeMap instead:\n${hashIterationViolations.join('\n')}`,
);
assert.ok(inspectedHashBindings >= 1, 'the hash-binding detector must still find the containers it is meant to guard');

// The `determinism` helpers themselves *do* call `HashMap::iter()`, which is
// exactly what the rule above forbids elsewhere.  That is safe because each
// helper materialises the pairs and sorts them before returning, so no caller
// ever observes hash order.  Asserting the sort (rather than skipping the file)
// keeps the exemption honest: strip the sort and this test fails.
for (const helper of ['sorted_pairs', 'sorted_keys', 'into_sorted_pairs']) {
  const body = determinismCode.match(new RegExp(`fn ${helper}<[\\s\\S]*?\\n\\{([\\s\\S]*?)\\n\\}`))?.[1];
  assert.ok(body, `determinism.rs must keep ${helper} as the sanctioned way to walk a hash container`);
  assert.match(
    body,
    /\.(?:iter|keys|into_iter)\(\)/,
    `${helper} must be the place where hash order is read, so nobody else has to`,
  );
  assert.match(
    body,
    /\.sort_unstable(?:_by)?\(/,
    `${helper} must sort before returning, otherwise it leaks hash order`,
  );
}
assert.match(
  determinism,
  /f64::total_cmp/,
  'the determinism module must name total_cmp as the required float comparison',
);

// --- 2. the reduction block length is a compile-time literal ---------------
// `rayon`'s own `sum()` cuts the input into as many pieces as the pool has work
// for, so its reduction tree — and the last mantissa bits of the result — moves
// with RAYON_NUM_THREADS.  A literal block length fixes the tree; deriving it
// from the thread count or the input length would put the thread count straight
// back into the output.

const literalUsizeConstant = (code, name) => {
  const declaration = code.match(new RegExp(`const ${name}\\s*:\\s*usize\\s*=\\s*([^;]+);`))?.[1]?.trim();
  assert.ok(declaration, `determinism.rs must declare ${name} as an explicit usize constant`);
  assert.match(
    declaration,
    /^[0-9][0-9_]*$/,
    `${name} must be a plain numeric literal, not derived from any runtime quantity (found "${declaration}")`,
  );
  return Number(declaration.replaceAll('_', ''));
};

const deterministicSumBlockLen = literalUsizeConstant(determinismCode, 'DETERMINISTIC_SUM_BLOCK_LEN');
assert.equal(
  deterministicSumBlockLen,
  4096,
  'the accumulation block length is part of the output bits: changing it must be a deliberate design change',
);
literalUsizeConstant(determinismCode, 'DETERMINISTIC_SUM_MIN_BLOCKS_PER_TASK');
assert.doesNotMatch(
  determinismCode,
  /current_num_threads/,
  'the reduction shape must not read the thread count, in any form',
);
assert.doesNotMatch(
  determinismCode,
  /num_threads\s*\(|available_parallelism/,
  'the reduction shape must not read the available parallelism either',
);
assert.match(
  determinismCode,
  /par_chunks\(DETERMINISTIC_SUM_BLOCK_LEN\)/,
  'the parallel path must cut the input at the constant block length, not at a runtime chosen width',
);
for (const helper of ['deterministic_sum', 'deterministic_sum_map', 'deterministic_mean']) {
  assert.match(
    determinismCode,
    new RegExp(`fn ${helper}[(<]`),
    `${helper} must stay available as the deterministic replacement for a rayon sum`,
  );
}

// --- 3. the RANSAC seed comes from neither the clock nor the thread id ------
// Every seeding site mixes a per-site compile-time constant with the run seed.
// The argument of each `seed_from_u64` is checked as a whole, because that is
// the only expression that can reach the sampling sequence.  This is narrower
// than banning the identifiers file wide on purpose: `panorama_stitching.rs`
// legitimately calls `Instant::now()` for its timing logs, and `report.rs`
// legitimately calls `SystemTime::now()` in `epoch_millis` for the report
// timestamps.  Neither may become a seed.

const callArguments = (code, callee) => {
  const needle = `${callee}(`;
  const argumentLists = [];
  let start = code.indexOf(needle);
  while (start !== -1) {
    let cursor = start + needle.length - 1;
    let depth = 0;
    for (; cursor < code.length; cursor += 1) {
      if (code[cursor] === '(') depth += 1;
      else if (code[cursor] === ')') {
        depth -= 1;
        if (depth === 0) break;
      }
    }
    argumentLists.push(code.slice(start + needle.length, cursor));
    start = code.indexOf(needle, cursor);
  }
  return argumentLists;
};

const NON_REPRODUCIBLE_SEED_INGREDIENTS = [
  'thread_rng',
  'from_entropy',
  'SystemTime',
  'Instant::now',
  'elapsed',
  'thread::current',
  'ThreadId',
  'current_num_threads',
  'rand::random',
];

const seedingSites = [
  ['determinism.rs', determinismCode],
  ['processing.rs', processingCode],
  ['panorama_stitching.rs', panoramaCode],
].flatMap(([file, code]) =>
  callArguments(code, 'seed_from_u64').map((argument) => ({ file, argument: argument.replace(/\s+/g, ' ').trim() })),
);
assert.ok(
  seedingSites.length >= 3,
  `the RANSAC seeding sites must stay visible to this guard, found ${seedingSites.length}`,
);
for (const { file, argument } of seedingSites) {
  for (const ingredient of NON_REPRODUCIBLE_SEED_INGREDIENTS) {
    assert.ok(
      !argument.includes(ingredient),
      `${file} seeds a generator from ${ingredient}: "${argument}" — seeds must be derived from input content only`,
    );
  }
}

const runSeedMixingSites = seedingSites.filter(({ argument }) => argument.includes('run_random_seed_mix()'));
assert.ok(
  runSeedMixingSites.length >= 3,
  `every RANSAC site must mix the content derived run seed, found ${runSeedMixingSites.length}`,
);
for (const { file, argument } of runSeedMixingSites) {
  assert.match(
    argument,
    /0x[0-9A-Fa-f_]+u64|\bseed\b/,
    `${file} must keep a per-site compile-time constant next to the run seed: "${argument}"`,
  );
}
for (const [file, code] of [
  ['determinism.rs', determinismCode],
  ['processing.rs', processingCode],
  ['panorama_stitching.rs', panoramaCode],
]) {
  assert.doesNotMatch(code, /thread_rng|from_entropy/, `${file} must not create an entropy seeded generator anywhere`);
}
assert.doesNotMatch(determinismCode, /SystemTime|Instant::now/, 'the seed derivation must never read a clock');
assert.match(
  stackReport,
  /fn epoch_millis\(\)[\s\S]*?SystemTime::now/,
  'report.rs keeps the only sanctioned clock read (report timestamps); this guard must stay aware of it',
);

// --- 4. the run seed is derived from the input and reported -----------------

assert.match(
  determinismCode,
  /fn derive_run_seed_from_paths\(pipeline_version: &str, source_paths: &\[String\]\) -> u64/,
  'the run seed must be derived from the pipeline version and the source paths',
);
assert.match(
  determinismCode,
  /sorted\.sort_unstable\(\)/,
  'the seed derivation must sort the paths, so the import order cannot reach the seed',
);
assert.match(determinismCode, /Sha256::new\(\)/, 'the run seed must be a SHA-256 digest of the input identity');
for (const helper of [
  'set_run_random_seed',
  'clear_run_random_seed',
  'run_random_seed_mix',
  'run_random_seed_source',
]) {
  assert.match(determinismCode, new RegExp(`fn ${helper}\\(`), `determinism.rs must expose ${helper}`);
}

const seedSourceIdentifiers = ['site_constants_without_run_seed', 'sha256_cache_key'];
for (const identifier of seedSourceIdentifiers) {
  assert.ok(
    determinism.includes(`"${identifier}"`),
    `resources.random_seed_source must keep the stable identifier "${identifier}"`,
  );
  assert.match(identifier, identifierPattern, `"${identifier}" must stay machine readable like every other identifier`);
}
assert.match(
  stackReport,
  /pub random_seed_source: String/,
  'the report must record how the RANSAC seed was derived, not only the thread count',
);
assert.match(
  panoramaCode,
  /random_seed_source = determinism::run_random_seed_source\(\)/,
  'the reported seed source must come from the installed seed rather than a hard coded string',
);
assert.match(
  panoramaCode,
  /set_run_random_seed\(determinism::derive_run_seed_from_cache_key\(/,
  'a focus stack run must install a seed derived from its own Virtual_Tile cache key',
);
assert.match(
  determinismCode,
  /fn derive_run_seed_from_cache_key\(cache_key: &\[u8; 32\]\) -> u64/,
  'the production run seed must be derived from the Virtual_Tile cache key',
);
assert.match(
  panoramaCode,
  /fn focus_stack_cache_key_inputs\(image_paths: &\[String\]\) -> FocusStackSourceDigests/,
  'the cache key ingredients must be assembled from the run selection',
);
assert.match(
  panoramaCode,
  /struct FocusStackSourceDigests \{[\s\S]*?inputs: virtual_tile::CacheKeyInputs/,
  'the run selection digests must carry the three cache key ingredients',
);

// ---------------------------------------------------------------------------
// Read-only Source_RAW access and the before/after digest check
// (requirements 4.8 / 12.8, asserted via 15.9)
// ---------------------------------------------------------------------------

assert.match(
  panoramaCode,
  /struct FocusStackSourceDigests \{[\s\S]*?sha256_before: Vec<String>/,
  'the digests taken before the run must be kept per source for the Stack_Report',
);
assert.match(
  panoramaCode,
  /with_source_digest_recheck\(Box::new\(\|paths\| \{[\s\S]*?focus_stack_source_digests\(paths\)/,
  'the recorder must re-read the sources on the terminating path, not trust the digests it already has',
);
assert.match(
  virtualTile,
  /fn source_file_sha256\(path: &Path\)[\s\S]*?File::open\(path\)/,
  'Source_RAW digests must be taken through a read-only File::open (requirement 4.8)',
);
assert.match(
  stackReport,
  /fn recheck_source_digests\(&self\)/,
  'report.rs must own the before/after comparison so every terminating path performs it',
);
const writeOnce = stackReport.match(/fn write_once\(&self\) -> Option<PathBuf> \{([\s\S]*?)\n    \}/)?.[1];
assert.ok(writeOnce, 'report.rs must keep write_once as the single serialisation point');
assert.match(
  writeOnce,
  /self\.recheck_source_digests\(\)/,
  'the digest re-check must run inside the write that every terminating path reaches',
);
assert.match(
  stackReport,
  /impl Drop for StackReportRecorder[\s\S]*?self\.write_once\(\)/,
  'the drop guard must keep covering the failure and cancellation paths',
);

// ---------------------------------------------------------------------------
// Virtual_Tile residency leases (requirement 14.4, asserted via 15.9)
// ---------------------------------------------------------------------------

assert.match(
  virtualTile,
  /MAX_RESIDENT_VIRTUAL_TILES: usize = 2/,
  'the two-tile residency ceiling of requirement 14.4 must stay an explicit constant',
);
assert.match(
  virtualTile,
  /fn lease\(\s*&self,\s*station_index: usize,/,
  'the store must hand out tiles through lease(station_index, ..) so residency is accounted for',
);
assert.match(
  virtualTile,
  /impl Drop for TileLease[\s\S]*?self\.registry\.release\(self\.station_index\)/,
  'a lease must return its slot by RAII rather than by an explicit release call',
);
assert.match(
  virtualTile,
  /ResidencyExhausted \{ limit: usize, held: usize \}/,
  'a residency request that cannot evict anything must fail with a measured error',
);
assert.match(
  virtualTile,
  /fn max_resident_virtual_tiles\(&self\) -> usize/,
  'the observed residency peak must be readable for resources.max_resident_virtual_tiles',
);
assert.match(
  stackReport,
  /fn record_max_resident_virtual_tiles\(&self, observed: usize\)/,
  'the report must accept the observed residency peak',
);
assert.match(
  stackReport,
  /pub max_resident_virtual_tiles: usize/,
  'resources must keep the observed maximum of simultaneously resident Virtual_Tiles',
);
assert.match(
  panoramaCode,
  /determinism::clear_run_random_seed\(\)/,
  'a run without a derived seed must clear the previous one instead of inheriting it',
);

console.log(
  `Validated focus-stack quality guards: ${selectionLongSide}-cell ownership grid, native refinement, ` +
    `station evidence (${stationMinInliers} inliers / ${stationMinOverlapNcc} overlap NCC / ` +
    `${stationMinSpatialSupport} spatial support / [${stationScaleRatioMin}, ${stationScaleRatioMax}] scale, ` +
    `${stationAccumulatedMotion} accumulated motion, ${stationMaxMembers} members), ` +
    'capture-group-first fusion, overlap-based global exposure, disagreement-aware seam cut, ' +
    'the setting-driven layered default compositor with no delivery crop and zero final sharpen, ' +
    `source-union crop, stable alignment regressions, ${identifierDeclarations.length} stable failure ` +
    `identifiers, the ${expectedTopLevelFields.length}-field Stack_Report schema, and the determinism ` +
    `baseline (${stackPipelineFiles.length} stack_pipeline files free of raw hash iteration, ` +
    `${deterministicSumBlockLen}-value literal reduction blocks, ${seedingSites.length} clock-free ` +
    'seeding sites).',
);
