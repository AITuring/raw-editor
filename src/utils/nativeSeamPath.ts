const median = (values: number[]) => [...values].sort((a, b) => a - b)[Math.floor(values.length / 2)] ?? 0;
const mean = (values: number[]) => values.reduce((a, b) => a + b, 0) / values.length;
const extent = 144;
const search = 48;
const count = search * 2 + 1;
const clip = (value: number) => 6 * Math.tanh(value / 6);
interface Strip {
  values: (number | null)[];
  noise: number;
  bias: number;
  wideNoise: number;
  wideBias: number;
}

export interface NativeSeamScan {
  training: (number | null)[];
  validation: (number | null)[];
  delta: (number | null)[];
  plateau: (number | null)[];
}

/** Full response curves, including weak responses rejected by local decisions. */
export function scanNativeSeam(
  image: ImageData,
  cx: number,
  cy: number,
  normal: [number, number],
  span = 480,
): NativeSeamScan {
  const strips = sampleStrips(image, cx, cy, normal, span);
  const result: NativeSeamScan = { training: [], validation: [], delta: [], plateau: [] };
  for (let offset = -112; offset <= 112; offset++) {
    const train: number[] = [],
      validate: number[] = [],
      delta: number[] = [],
      plateau: number[] = [];
    for (let i = 0; i < strips.length; i++) {
      const response = score(strips[i], offset);
      if (response === null) continue;
      if (i % 2 === 0) train.push(response);
      else {
        validate.push(response);
        const d = difference(strips[i], offset, 4);
        if (d !== null) delta.push(d);
        const values = strips[i].values;
        const a = values.slice(extent + offset - 16, extent + offset - 8),
          b = values.slice(extent + offset + 8, extent + offset + 16);
        if (![...a, ...b].some((v) => v === null)) plateau.push(mean(b as number[]) - mean(a as number[]));
      }
    }
    result.training.push(train.length >= 6 ? mean(train) : null);
    result.validation.push(validate.length >= 6 ? mean(validate) : null);
    result.delta.push(delta.length >= 6 ? median(delta) : null);
    result.plateau.push(plateau.length >= 6 ? median(plateau) : null);
  }
  return result;
}

export interface NativeSeamSequence {
  status: 'stable' | 'insufficient';
  offsets: number[];
  delta: number;
  support: number;
  samples: number;
  trainingScore: number;
  validationScore: number;
  controlScore: number;
}

/** Path optimization uses training strips only. Every retained path, including
 * weak ones, faces the same held-out and translated same-side control tests. */
function resolveSequences(
  scans: NativeSeamScan[],
  normalOrigins?: number[],
): { best: NativeSeamSequence; accepted: NativeSeamSequence[]; pending: NativeSeamSequence[] } {
  const empty: NativeSeamSequence = {
    status: 'insufficient',
    offsets: [],
    delta: 0,
    support: 0,
    samples: scans.length,
    trainingScore: 0,
    validationScore: 0,
    controlScore: 0,
  };
  if (scans.length < 6) return { best: empty, accepted: [], pending: [] };
  const hypotheses: { offsets: number[]; sign: number; total: number; straight?: boolean }[] = [];
  // State includes path slope. Penalize changes in slope, not every normal
  // displacement: a weak slanted/curved seam must not lose to a straight texture.
  const slopeCount = 33;
  for (const sign of [-1, 1]) {
    let cost = new Float64Array(count * slopeCount);
    const parents: Int16Array[] = [];
    for (let row = 0; row < scans.length; row++) {
      const next = new Float64Array(cost.length),
        parent = new Int16Array(cost.length);
      next.fill(-Infinity);
      parent.fill(-1);
      const shift = row && normalOrigins ? normalOrigins[row] - normalOrigins[row - 1] : 0;
      for (let k = 0; k < count; k++) {
        const value = scans[row].training[k - search + 112];
        const emission = value === null ? -0.1 : value * sign;
        for (let d = 0; d < slopeCount; d++) {
          const state = k * slopeCount + d;
          if (!row) {
            next[state] = emission;
            continue;
          }
          const prior = Math.round(k + shift - (d - 16));
          if (prior < 0 || prior >= count) continue;
          for (let old = 0; old < slopeCount; old++) {
            const priorState = prior * slopeCount + old;
            const candidate = cost[priorState] + emission - 0.04 * Math.abs(old - d) - 0.01 * Math.abs(d - 16);
            if (candidate > next[state]) {
              next[state] = candidate;
              parent[state] = priorState;
            }
          }
        }
      }
      cost = next;
      parents.push(parent);
    }
    const ends = Array.from({ length: cost.length }, (_, i) => i).sort((a, b) => cost[b] - cost[a]);
    const retained: number[] = [];
    for (const end of ends) {
      const position = Math.floor(end / slopeCount);
      if (!Number.isFinite(cost[end]) || retained.some((p) => Math.abs(p - position) < 6)) continue;
      retained.push(position);
      let state = end;
      const offsets = new Array<number>(scans.length);
      for (let row = scans.length - 1; row >= 0; row--) {
        offsets[row] = Math.floor(state / slopeCount) - search;
        state = parents[row][state];
      }
      hypotheses.push({ offsets, sign, total: cost[end] });
      if (retained.length === 16) break;
    }
  }
  // A stronger nearby contour must not consume every curved-path hypothesis.
  // Preserve straight paths as separate models before any per-window winner.
  for (let offset = -search; offset <= search; offset++)
    for (const sign of [-1, 1]) {
      const offsets = scans.map((_, i) =>
        Math.round(offset - (normalOrigins ? normalOrigins[i] - normalOrigins[0] : 0)),
      );
      if (offsets.some((o) => Math.abs(o) > search)) continue;
      const values = offsets.map((o, i) => scans[i].training[o + 112]);
      if (values.filter((v) => v !== null).length < 6) continue;
      hypotheses.push({
        offsets,
        sign,
        straight: true,
        total: values.reduce<number>((sum, v) => sum + (v === null ? -0.1 : v * sign), 0),
      });
    }
  const accepted: NativeSeamSequence[] = [],
    pending: NativeSeamSequence[] = [],
    straightPending: NativeSeamSequence[] = [];
  let result = empty;
  for (const h of hypotheses) {
    const values = h.offsets.map((offset, i) => scans[i].validation[offset + 112]);
    const validIndices = values.flatMap((v, i) => (v === null ? [] : [i]));
    if (validIndices.length < 6) continue;
    const validation = mean(validIndices.map((i) => values[i]!)) * h.sign * Math.sqrt(validIndices.length);
    const training = h.total / Math.sqrt(scans.length);
    const deltas = h.offsets.map((offset, i) => scans[i].delta[offset + 112]).filter((v): v is number => v !== null);
    const plateaus = h.offsets
      .map((offset, i) => scans[i].plateau[offset + 112])
      .filter((v): v is number => v !== null);
    const signal = median(deltas);
    const support = deltas.filter((v) => v * h.sign > Math.max(0.005, Math.abs(signal) * 0.25)).length;
    const controls: number[] = [];
    for (const shift of [-64, -56, -48, -40, -32, -24, -16, 16, 24, 32, 40, 48, 56, 64]) {
      const v = validIndices.map((i) => scans[i].validation[h.offsets[i] + shift + 112]);
      if (v.some((x) => x === null || x === undefined)) continue;
      controls.push(mean(v as number[]) * h.sign * Math.sqrt(validIndices.length));
    }
    const control = Math.max(0, ...controls),
      delta = median(deltas);
    const center = median(controls);
    const uncertainty = Math.max(0.12, 1.4826 * median(controls.map((v) => Math.abs(v - center))));
    const boundaryHits = h.offsets.filter((offset) => Math.abs(offset) >= search - 2).length;
    const stable =
      boundaryHits < Math.ceil(scans.length * 0.25) &&
      training > center + 5 * uncertainty &&
      validation > center + 5 * uncertainty &&
      validation > control + uncertainty &&
      controls.length >= 10 &&
      support >= Math.ceil(validIndices.length * 0.75) &&
      deltas.length >= 6 &&
      plateaus.length >= 6 &&
      delta * h.sign >= 0.025 &&
      median(plateaus) * h.sign >= Math.abs(delta) * 0.4;
    const evidence: NativeSeamSequence = {
      status: stable ? 'stable' : 'insufficient',
      offsets: h.offsets,
      delta,
      support,
      samples: scans.length,
      trainingScore: training,
      validationScore: validation,
      controlScore: control,
    };
    if (boundaryHits < Math.ceil(scans.length * 0.25)) {
      pending.push(evidence);
      if (h.straight) straightPending.push(evidence);
    }
    if (stable) {
      const candidate: NativeSeamSequence = {
        status: 'stable',
        offsets: h.offsets,
        delta,
        support,
        samples: scans.length,
        trainingScore: training,
        validationScore: validation,
        controlScore: control,
      };
      accepted.push(candidate);
      if (result.status !== 'stable' || training > result.trainingScore) result = candidate;
    } else if (result.status !== 'stable' && validation > result.validationScore)
      result = {
        status: 'insufficient',
        offsets: h.offsets,
        delta,
        support,
        samples: scans.length,
        trainingScore: training,
        validationScore: validation,
        controlScore: control,
      };
  }
  const distinct: NativeSeamSequence[] = [];
  for (const candidate of accepted.sort((a, b) => b.trainingScore - a.trainingScore)) {
    if (!distinct.some((c) => c.offsets.every((o, i) => Math.abs(o - candidate.offsets[i]) < 6)))
      distinct.push(candidate);
  }
  const seeds: NativeSeamSequence[] = [];
  for (const c of pending.sort((a, b) => b.trainingScore - a.trainingScore)) {
    if (!seeds.some((p) => p.offsets.every((o, i) => Math.abs(o - c.offsets[i]) < 6))) seeds.push(c);
    if (seeds.length === 6) break;
  }
  // Keep straight and curved model families independently at proposal time.
  // A high-scoring curved contour must not remove a nearby weak straight seam
  // before its native-resolution evidence has even been examined.
  for (const sign of [-1, 1]) {
    const straight: NativeSeamSequence[] = [];
    for (const c of straightPending
      .filter((c) => c.delta * sign > 0)
      .sort((a, b) => b.trainingScore - a.trainingScore)) {
      if (straight.some((p) => p.offsets.every((o, i) => Math.abs(o - c.offsets[i]) < 6))) continue;
      straight.push(c);
      if (!seeds.some((p) => p.offsets.every((o, i) => Math.abs(o - c.offsets[i]) < 6))) seeds.push(c);
      if (straight.length === 2) break;
    }
  }
  return { best: result, accepted: distinct, pending: seeds };
}

export function resolveNativeSeamSequence(scans: NativeSeamScan[], normalOrigins?: number[]): NativeSeamSequence {
  return resolveSequences(scans, normalOrigins).best;
}
export function resolveNativeSeamSequences(
  scans: NativeSeamScan[],
  normalOrigins?: number[],
  proposalsOnly = false,
): NativeSeamSequence[] {
  const result = resolveSequences(scans, normalOrigins);
  return proposalsOnly ? result.pending : result.accepted;
}

function difference(strip: Strip, offset: number, width = 2): number | null {
  const left = strip.values.slice(extent + offset - width, extent + offset);
  const right = strip.values.slice(extent + offset, extent + offset + width);
  if (left.length !== width || right.length !== width || [...left, ...right].some((v) => v === null)) return null;
  return mean(right as number[]) - mean(left as number[]);
}

function score(strip: Strip, offset: number): number | null {
  const fine = difference(strip, offset, 2),
    wide = difference(strip, offset, 8);
  return fine === null || wide === null
    ? null
    : (clip((fine - strip.bias) / strip.noise) + clip((wide - strip.wideBias) / strip.wideNoise)) / 2;
}

function sampleStrips(image: ImageData, cx: number, cy: number, normal: [number, number], span = 480): Strip[] {
  const length = Math.hypot(...normal);
  if (!Number.isFinite(length) || !length) return [];
  const nx = normal[0] / length,
    ny = normal[1] / length,
    tx = -ny,
    ty = nx;
  const spacing = span / 16,
    half = Math.max(0, Math.floor(spacing * 0.4));
  return Array.from({ length: 16 }, (_, index) => {
    const values: (number | null)[] = [];
    for (let n = -extent; n <= extent; n++) {
      const samples: number[] = [];
      for (let t = -half; t <= half; t++) {
        const tangent = (index - 7.5) * spacing + t;
        const x = Math.round(cx + n * nx + tangent * tx),
          y = Math.round(cy + n * ny + tangent * ty);
        if (x < 0 || y < 0 || x >= image.width || y >= image.height) continue;
        const i = (y * image.width + x) * 4;
        const light = 0.2126 * image.data[i] + 0.7152 * image.data[i + 1] + 0.0722 * image.data[i + 2];
        if (image.data[i + 3] >= 250 && light > 12 && light < 240) samples.push(Math.log2(light));
      }
      samples.sort((a, b) => a - b);
      values.push(
        samples.length >= Math.ceil((half * 2 + 1) * 0.84)
          ? mean(samples.slice(Math.floor(samples.length * 0.12), samples.length - Math.floor(samples.length * 0.12)))
          : null,
      );
    }
    const strip: Strip = { values, noise: 1, bias: 0, wideNoise: 1, wideBias: 0 };
    const controls: number[] = [];
    for (let offset = -112; offset <= 112; offset += 2) {
      const value = difference(strip, offset);
      if (value !== null) controls.push(value);
    }
    strip.bias = median(controls);
    strip.noise = Math.max(0.002, 1.4826 * median(controls.map((v) => Math.abs(v - strip.bias))));
    const wide: number[] = [];
    for (let offset = -112; offset <= 112; offset += 2) {
      const v = difference(strip, offset, 8);
      if (v !== null) wide.push(v);
    }
    strip.wideBias = median(wide);
    strip.wideNoise = Math.max(0.002, 1.4826 * median(wide.map((v) => Math.abs(v - strip.wideBias))));
    return strip;
  });
}
