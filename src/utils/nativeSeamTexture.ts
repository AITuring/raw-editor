import type { TextureStatus } from './seamTexture';

export interface NativeTextureEvidence {
  status: TextureStatus;
  crossDifference: number;
  sameSideDifference: number;
  excessDifference: number;
  scales: number;
  /** Spatially separate tangent bands used for the decision. */
  bands?: number;
  supportedBands?: number;
  inconsistentBands?: number;
  /** No resampling: all distances below are in original source pixels. */
  nativePixels: true;
}
const median = (a: number[]) => [...a].sort((x, y) => x - y)[Math.floor(a.length / 2)] ?? 0;
interface Descriptor {
  energy: number;
  bins: number[];
}

/** Distributional descriptors deliberately do not require adjacent random fibres
 * to have the same phase. Compare equally spaced within-side controls instead. */
function examineBand(image: ImageData, cx: number, cy: number, normal: [number, number]): NativeTextureEvidence {
  const length = Math.hypot(...normal),
    nx = normal[0] / length,
    ny = normal[1] / length,
    tx = -ny,
    ty = nx;
  const empty: NativeTextureEvidence = {
    status: 'insufficient',
    crossDifference: 0,
    sameSideDifference: 0,
    excessDifference: 0,
    scales: 0,
    nativePixels: true,
  };
  if (!length) return empty;
  const light = (x: number, y: number) => {
    x = Math.round(x);
    y = Math.round(y);
    if (x < 0 || y < 0 || x >= image.width || y >= image.height) return null;
    const i = (y * image.width + x) * 4;
    const l = 0.2126 * image.data[i] + 0.7152 * image.data[i + 1] + 0.0722 * image.data[i + 2];
    return image.data[i + 3] >= 250 && l > 12 && l < 240 ? Math.log2(l) : null;
  };
  const descriptor = (offset: number, scale: number): Descriptor | null => {
    const bins = new Array<number>(8).fill(0);
    let energy = 0,
      count = 0;
    const derivatives: { gn: number; gt: number }[] = [];
    for (let t = -24; t <= 24; t += 2)
      for (let n = -6; n <= 6; n++) {
        const x = cx + (offset + n) * nx + t * tx,
          y = cy + (offset + n) * ny + t * ty;
        const a = light(x - scale * nx, y - scale * ny),
          b = light(x + scale * nx, y + scale * ny);
        const c = light(x - scale * tx, y - scale * ty),
          d = light(x + scale * tx, y + scale * ty);
        if (a === null || b === null || c === null || d === null) continue;
        const gn = (b - a) / (2 * scale),
          gt = (d - c) / (2 * scale);
        count++;
        derivatives.push({ gn, gt });
      }
    if (count < 260) return null;
    // A local ink/shading slope is not the material texture. Remove a robust
    // plane before comparing the residual grain on either side.
    const trendN = median(derivatives.map((g) => g.gn)),
      trendT = median(derivatives.map((g) => g.gt));
    const gradients = derivatives.map((g) => {
      const gn = g.gn - trendN,
        gt = g.gt - trendT;
      const angle = (Math.atan2(gt, gn) + Math.PI) % Math.PI;
      return { e: gn * gn + gt * gt, bin: Math.min(7, Math.floor((angle / Math.PI) * 8)) };
    });
    // Limit the influence of sparse ink contours while retaining the material's
    // gradient distribution. A blank block still has zero energy.
    const cap = gradients.map((g) => g.e).sort((a, b) => a - b)[Math.floor(gradients.length * 0.75)];
    for (const g of gradients) {
      const e = Math.min(g.e, cap);
      energy += e;
      bins[g.bin] += e;
    }
    return { energy: energy / count, bins: bins.map((v) => v / Math.max(1e-12, energy)) };
  };
  const distance = (a: Descriptor, b: Descriptor) =>
    Math.abs(Math.log2(Math.max(a.energy, 1e-8) / Math.max(b.energy, 1e-8))) +
    a.bins.reduce((s, v, i) => s + Math.abs(v - b.bins[i]), 0);
  const crosses: number[] = [],
    baselines: number[] = [],
    votes: TextureStatus[] = [];
  for (const scale of [1, 2, 4]) {
    // Cross seam pair and both control pairs have equal 32-pixel spacing.
    const blocks = [-80, -48, -16, 16, 48, 80].map((n) => descriptor(n, scale));
    if (blocks.some((v) => !v)) continue;
    const b = blocks as Descriptor[];
    // Derivatives were divided by the sampling distance above. Compare a
    // fixed contrast floor in intensity-difference units, not derivative units.
    if (b.some((v) => v.energy * scale * scale < 0.000015)) continue;
    const control = [distance(b[0], b[1]), distance(b[1], b[2]), distance(b[3], b[4]), distance(b[4], b[5])];
    const baseline = median(control),
      cross = distance(b[2], b[3]);
    crosses.push(cross);
    baselines.push(baseline);
    // Heterogeneous controls cannot establish a useful reference distribution.
    if (Math.max(...control) > 2.5 || baseline > 1.3) votes.push('insufficient');
    else if (cross > Math.max(...control) + 0.8) votes.push('inconsistent');
    else if (cross <= Math.max(...control) + 0.25) votes.push('supported');
    else votes.push('insufficient');
  }
  const status: TextureStatus =
    votes.filter((v) => v === 'inconsistent').length >= 2
      ? 'inconsistent'
      : votes.filter((v) => v === 'supported').length >= 2
        ? 'supported'
        : 'insufficient';
  return {
    ...empty,
    status,
    crossDifference: median(crosses),
    sameSideDifference: median(baselines),
    excessDifference: median(crosses.map((v, i) => v - baselines[i])),
    scales: votes.length,
  };
}

/** Replicate the cross/same-side comparison along the seam. Spatially separate
 * bands reduce dependence on a single brush stroke; none is chosen for having
 * the best cross-seam score. Shorter normal distances keep controls local. */
export function examineNativeTexture(
  image: ImageData,
  cx: number,
  cy: number,
  normal: [number, number],
): NativeTextureEvidence {
  const length = Math.hypot(...normal);
  const empty: NativeTextureEvidence = {
    status: 'insufficient',
    crossDifference: 0,
    sameSideDifference: 0,
    excessDifference: 0,
    scales: 0,
    nativePixels: true,
    bands: 0,
    supportedBands: 0,
    inconsistentBands: 0,
  };
  if (!Number.isFinite(length) || !length) return empty;
  const tx = -normal[1] / length,
    ty = normal[0] / length;
  const bands = [-168, -112, -56, 0, 56, 112, 168].map((t) => examineBand(image, cx + t * tx, cy + t * ty, normal));
  const supported = bands.filter((b) => b.status === 'supported').length;
  const inconsistent = bands.filter((b) => b.status === 'inconsistent').length;
  const informative = bands.filter((b) => b.scales >= 2);
  const status: TextureStatus =
    inconsistent >= 3 ? 'inconsistent' : supported >= 5 && inconsistent === 0 ? 'supported' : 'insufficient';
  return {
    ...empty,
    status,
    crossDifference: median(informative.map((b) => b.crossDifference)),
    sameSideDifference: median(informative.map((b) => b.sameSideDifference)),
    excessDifference: median(informative.map((b) => b.excessDifference)),
    scales: informative.length >= 5 ? Math.min(...informative.map((b) => b.scales)) : 0,
    bands: informative.length,
    supportedBands: supported,
    inconsistentBands: inconsistent,
  };
}

export function summarizeNativeTexture(samples: NativeTextureEvidence[]): NativeTextureEvidence {
  const informative = samples.filter((s) => s.scales >= 2);
  const supported = samples.filter((s) => s.status === 'supported').length;
  const inconsistent = samples.filter((s) => s.status === 'inconsistent').length;
  return {
    status:
      samples.length >= 4 && supported / samples.length >= 0.65
        ? 'supported'
        : samples.length >= 4 && inconsistent / samples.length >= 0.45
          ? 'inconsistent'
          : 'insufficient',
    crossDifference: median(informative.map((s) => s.crossDifference)),
    sameSideDifference: median(informative.map((s) => s.sameSideDifference)),
    excessDifference: median(informative.map((s) => s.excessDifference)),
    scales: informative.length,
    nativePixels: true,
  };
}
