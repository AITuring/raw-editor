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

const numberConstant = (source, name) => {
  const value = source.match(new RegExp(`(?:const|pub(?:\\([^)]*\\))? const)\\s+${name}[^=]*=\\s*([0-9]+(?:\\.[0-9]+)?)`))?.[1];
  assert.ok(value, `${name} must remain an explicit numeric quality guard`);
  return Number(value);
};

const selectionLongSide = numberConstant(mosaic, 'SELECTION_LONG_SIDE');
const nativeRefineStep = numberConstant(mosaic, 'NATIVE_REFINE_STEP');
const mismatchPenalty = numberConstant(mosaic, 'OWNERSHIP_MISMATCH_PENALTY');
assert.ok(selectionLongSide >= 512, 'ownership grid must remain fine enough for thin brush strokes');
assert.ok(nativeRefineStep <= 16, 'native refinement must sample at a sufficiently fine spacing');
assert.ok(mismatchPenalty >= 1, 'misregistered sharp candidates need a non-trivial mismatch penalty');

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
assert.match(
  stitching,
  /FOCUS_ALLOW_LOW_FREQUENCY_SEAM_BLEND/,
  'seam colour correction must remain explicitly gated',
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
  'large stacks must validate real image agreement across camera-position boundaries',
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
assert.match(panorama, /focus_alignment_does_not_trade_fit_precision_for_a_few_more_inliers/, 'fit selection must protect precision over raw inlier count');
assert.match(panorama, /focus_stack_stability_rejects_extrapolated_projective_warp/, 'extrapolated projective warps need a regression test');

console.log(
  `Validated focus-stack quality guards: ${selectionLongSide}-cell ownership grid, native refinement, ` +
    'capture-group-first fusion, overlap-based global exposure, disagreement-aware seam cut, ' +
    'source-union crop, and stable alignment regressions.',
);
