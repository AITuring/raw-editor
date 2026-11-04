/** Original-pixel edge evidence. A stable step is not proof of a stitching defect:
 * paint and shadows can produce the same signal in a flattened photograph. */
export interface NativeSeamProfile {
  status: 'stable' | 'insufficient';
  offset: number;
  delta: number;
  validationSupport: number;
  validationSamples: number;
  locationSpread: number;
  controlRatio: number;
}

const median = (values: number[]) => {
  const sorted = [...values].sort((a, b) => a - b);
  return sorted.length ? sorted[Math.floor(sorted.length / 2)] : 0;
};
const radius = 192;
const mean = (values: number[]) => values.reduce((sum, v) => sum + v, 0) / values.length;
const trimmedMean = (values: number[]) => {
  const sorted = [...values].sort((a, b) => a - b),
    trim = Math.floor(sorted.length * 0.15);
  return mean(sorted.slice(trim, sorted.length - trim));
};
interface Step {
  offset: number;
  delta: number;
  score: number;
  residual: number;
  sharpness: number;
}

/** Compare near bands after subtracting the slope estimated on each same side.
 * This measures the discontinuity locally without fitting across painted detail. */
function contrast(profile: (number | null)[], offset: number, width: number, detrend = true): number | null {
  const band = (start: number) => {
    const values = Array.from({ length: width }, (_, i) => profile[radius + offset + start + i]);
    return values.every((v) => v != null) ? mean(values as number[]) : null;
  };
  const left = band(-width),
    right = band(0);
  const farLeft = band(-3 * width),
    farRight = band(2 * width);
  if (left === null || right === null || farLeft === null || farRight === null) return null;
  const slopeContribution = (left - farLeft + (farRight - right)) / 4;
  return right - left - (detrend ? slopeContribution : 0);
}

function stepAt(profile: (number | null)[], offset: number): Step | null {
  const narrow = contrast(profile, offset, 4),
    wide = contrast(profile, offset, 16);
  if (narrow === null || wide === null) return null;
  const direct = contrast(profile, offset, 4, false)!;
  // Slope subtraction alone creates side lobes next to a thin painted line.
  if (direct * Math.sign(narrow) < Math.abs(narrow) * 0.45) return null;
  const farLeft = profile.slice(radius + offset - 32, radius + offset - 16);
  const farRight = profile.slice(radius + offset + 16, radius + offset + 32);
  if ([...farLeft, ...farRight].some((v) => v == null)) return null;
  const plateau = mean(farRight as number[]) - mean(farLeft as number[]);
  if (plateau * Math.sign(narrow) < Math.abs(narrow) * 0.5) return null;
  const controls = [-96, -64, -32, 32, 64, 96].map((n) => contrast(profile, offset + n, 4));
  if (controls.some((v) => v === null)) return null;
  const residual = median(controls.map((v) => Math.abs(v!)));
  return {
    offset,
    delta: narrow,
    residual,
    score: Math.abs(narrow) - residual,
    sharpness: (Math.sign(wide) * narrow) / Math.max(Math.abs(wide), 0.001),
  };
}

function plausible(step: Step | null): step is Step {
  return (
    !!step &&
    Math.abs(step.delta) >= 0.025 &&
    step.score > 0.003 &&
    Math.abs(step.delta) > 4 * step.residual &&
    step.sharpness > 0.45 &&
    step.sharpness < 1.8
  );
}

export function examineNativeSeamProfile(
  image: ImageData,
  cx: number,
  cy: number,
  normal: [number, number],
): NativeSeamProfile {
  const empty: NativeSeamProfile = {
    status: 'insufficient',
    offset: 0,
    delta: 0,
    validationSupport: 0,
    validationSamples: 0,
    locationSpread: 0,
    controlRatio: 0,
  };
  const length = Math.hypot(...normal);
  if (!Number.isFinite(length) || !length) return empty;
  const nx = normal[0] / length,
    ny = normal[1] / length,
    tx = -ny,
    ty = nx;
  // Disjoint tangent strips: even strips choose position, odd strips validate it.
  const profiles = Array.from({ length: 8 }, (_, strip) => {
    const profile: (number | null)[] = [];
    for (let n = -radius; n <= radius; n++) {
      const values: number[] = [];
      for (let t = -20; t <= 20; t++) {
        const tangent = (strip - 3.5) * 48 + t;
        const x = Math.round(cx + n * nx + tangent * tx),
          y = Math.round(cy + n * ny + tangent * ty);
        if (x < 0 || y < 0 || x >= image.width || y >= image.height) continue;
        const i = (y * image.width + x) * 4;
        const lum = 0.2126 * image.data[i] + 0.7152 * image.data[i + 1] + 0.0722 * image.data[i + 2];
        if (image.data[i + 3] >= 250 && lum > 12 && lum < 240) values.push(Math.log2(lum));
      }
      profile.push(values.length >= 35 ? trimmedMean(values) : null);
    }
    return profile;
  });
  const combine = (indices: number[]) =>
    profiles[0].map((_, n) => {
      const values = indices.map((i) => profiles[i][n]).filter((v): v is number => v !== null);
      return values.length >= 3 ? mean(values) : null;
    });
  const training = combine([0, 2, 4, 6]);
  const candidates: Step[] = [];
  for (let offset = -48; offset <= 48; offset++) {
    const step = stepAt(training, offset);
    if (plausible(step)) candidates.push(step);
  }
  const best = candidates.sort((a, b) => b.score - a.score)[0];
  if (!best) return empty;
  const validation = combine([1, 3, 5, 7]);
  const heldOut = stepAt(validation, best.offset);
  if (!plausible(heldOut) || Math.sign(heldOut.delta) !== Math.sign(best.delta))
    return { ...empty, offset: best.offset, delta: best.delta };
  const pooledControls = [-80, 80].map((n) => contrast(validation, best.offset + n, 4));
  if (
    pooledControls.some((v) => v === null) ||
    Math.max(...pooledControls.map((v) => Math.abs(v!))) > Math.abs(heldOut.delta) * 0.5
  ) {
    return { ...empty, offset: best.offset, delta: best.delta };
  }
  let validationSamples = 0;
  const supported: Step[] = [],
    controlRatios: number[] = [];
  for (const i of [1, 3, 5, 7]) {
    const profile = profiles[i],
      at = stepAt(profile, best.offset);
    if (!at) continue;
    validationSamples++;
    if (
      Math.sign(at.delta) !== Math.sign(best.delta) ||
      Math.abs(at.delta - best.delta) > Math.max(0.025, Math.abs(best.delta) * 0.35)
    )
      continue;
    // Test equal windows away from the proposed edge, rather than rewarding
    // arbitrary strong transitions in a naturally heterogeneous patch.
    const controls = [-80, 80].map((n) => contrast(profile, best.offset + n, 4));
    if (controls.some((c) => c === null)) continue;
    const ratio = Math.max(...controls.map((c) => Math.abs(c!))) / Math.abs(at.delta);
    controlRatios.push(ratio);
    const local: Step[] = [];
    for (let shift = -6; shift <= 6; shift++) {
      const s = stepAt(profile, best.offset + shift);
      if (s && Math.sign(s.delta) === Math.sign(best.delta)) local.push(s);
    }
    const peak = local.sort((a, b) => b.score - a.score)[0];
    if (peak && Math.abs(peak.offset - best.offset) <= 4) supported.push(peak);
  }
  const offset = supported.length ? median(supported.map((s) => s.offset)) : best.offset;
  return {
    status: supported.length >= 3 ? 'stable' : 'insufficient',
    offset,
    delta: supported.length ? median(supported.map((s) => s.delta)) : best.delta,
    validationSupport: supported.length,
    validationSamples,
    locationSpread: supported.length ? Math.max(...supported.map((s) => Math.abs(s.offset - offset))) : 0,
    controlRatio: median(controlRatios),
  };
}

export interface NativeSeamEvidence {
  status: 'supported' | 'insufficient';
  samples: number;
  supported: number;
  delta: number;
}


export function summarizeNativeSeam(
  samples: { profile: NativeSeamProfile; textureSupported: boolean }[],
): NativeSeamEvidence {
  const accepted = samples.filter((s) => s.profile.status === 'stable' && s.textureSupported);
  const delta = median(accepted.map((s) => s.profile.delta));
  const consistent = accepted.filter(
    (s) =>
      Math.sign(s.profile.delta) === Math.sign(delta) &&
      Math.abs(s.profile.delta - delta) <= Math.max(0.025, Math.abs(delta) * 0.35),
  );
  return {
    status: samples.length >= 4 && consistent.length / samples.length >= 0.65 ? 'supported' : 'insufficient',
    samples: samples.length,
    supported: consistent.length,
    delta,
  };
}
