import type { NativeTextureEvidence } from './nativeSeamTexture';
import type { NativeSeamEvidence } from './nativeSeamProfile';
import type { NativeSeamSequence } from './nativeSeamPath';
import { examineSeamTexture, summarizeSeamTexture, type TextureEvidence } from './seamTexture';
/** Diagnostic candidates, not a semantic assertion that an edge is a stitching defect. */
export interface SeamCandidate {
  id: number;
  points: [number, number][];
  normals: [number, number][];
  nativeTexture?: NativeTextureEvidence;
  nativeProfile?: NativeSeamEvidence;
  nativeLocations?: [number, number][];
  nativeSequence?: NativeSeamSequence;
  nativePath?: [number,number][];
  proposal?: 'coherent';
  exposureDifference: number;
  confidence: number;
  texture: TextureEvidence;
  bounds: { x: number; y: number; width: number; height: number };
}
const median = (a: number[]) => { a.sort((x, y) => x - y); return a[Math.floor(a.length / 2)] ?? 0; };

/** Multi-scale paired samples reject ramps, chromatic object edges and thin dark lines.
 * Connected edge support allows curved and diagonal candidates, independent of any UI grid.
 * Exposure difference is in log2 encoded luminance, not a physical RAW exposure estimate.
 */
export function detectSeamCandidates(source: ImageData): SeamCandidate[] {
  const scale = Math.min(1, 1200 / Math.max(source.width, source.height));
  const width = Math.max(1, Math.round(source.width * scale));
  const height = Math.max(1, Math.round(source.height * scale));
  if (width < 16 || height < 16) return [];
  const light = new Float32Array(width * height), red = new Float32Array(width * height), blue = new Float32Array(width * height);
  const valid = new Uint8Array(width * height);
  for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
    const px = Math.min(source.width - 1, Math.floor((x + .5) / scale));
    const py = Math.min(source.height - 1, Math.floor((y + .5) / scale));
    const i = (py * source.width + px) * 4, j = y * width + x;
    const r = source.data[i], g = source.data[i + 1], b = source.data[i + 2];
    const l = .2126 * r + .7152 * g + .0722 * b;
    light[j] = Math.log2(Math.max(l, 1)); red[j] = r / Math.max(l, 1); blue[j] = b / Math.max(l, 1);
    valid[j] = source.data[i + 3] >= 250 && l > 16 && l < 235 ? 1 : 0;
  }
  const step = new Float32Array(width * height), quality = new Float32Array(width * height);
  const direction = new Int8Array(width * height);
  const normals = [[1, 0], [0, 1], [1, 1], [1, -1]];
  for (let y = 11; y < height - 11; y += 2) for (let x = 11; x < width - 11; x += 2) {
    let best = 0;
    for (let radius = 2; radius <= 4; radius++) for (let d = 0; d < normals.length; d++) {
      const [nx, ny] = normals[d], tx = -ny, ty = nx;
      const jumps: number[] = []; let colorDifference = 0;
      for (let t = -2; t <= 2; t++) {
        const at = (n: number) => (y + t * ty + n * ny) * width + x + t * tx + n * nx;
        const a = at(-radius), b = at(radius), c = at(-2 * radius), e = at(2 * radius);
        if (![a, b, c, e].every(i => valid[i])) continue;
        const near = light[b] - light[a], far = light[e] - light[c];
        // A ramp has far ~= 2*near. A thin scratch has little persistent far step.
        if (2 * Math.abs(near) - Math.abs(far) < .045 || near * far <= 0 || Math.abs(near) < .04 || Math.abs(near) < Math.abs(far) * .65 || Math.abs(near) > Math.abs(far) * 1.6) continue;
        jumps.push(near);
        colorDifference += Math.abs(red[b] - red[a]) + Math.abs(blue[b] - blue[a]);
      }
      if (jumps.length < 4 || colorDifference / jumps.length > .08) continue;
      const jump = median([...jumps]);
      const deviation = median(jumps.map(v => Math.abs(v - jump)));
      if (jumps.some(v => v * jump <= 0) || deviation > Math.abs(jump) * .25 || Math.abs(jump) > .8) continue;
      const score = Math.min(1, Math.abs(jump) / .18) * (1 - deviation / Math.max(.001, Math.abs(jump))) * (1 - colorDifference / 5 / .12);
      if (score > best) { best = score; const i = y * width + x; step[i] = jump; quality[i] = score; direction[i] = d; }
    }
  }
  const seen = new Uint8Array(width * height), candidates: SeamCandidate[] = [];
  for (let y = 11; y < height - 11; y += 2) for (let x = 11; x < width - 11; x += 2) {
    const start = y * width + x;
    if (seen[start] || !quality[start]) continue;
    const queue = [start]; seen[start] = 1;
    for (let q = 0; q < queue.length; q++) {
      const i = queue[q], cy = Math.floor(i / width), cx = i % width;
      for (let dy = -2; dy <= 2; dy += 2) for (let dx = -2; dx <= 2; dx += 2) {
        const px = cx + dx, py = cy + dy, j = py * width + px;
        if (px < 11 || py < 11 || px >= width - 11 || py >= height - 11 || seen[j] || !quality[j]) continue;
        const [ax, ay] = normals[direction[i]], [bx, by] = normals[direction[j]];
        if (Math.abs(ax * bx + ay * by) / Math.hypot(ax, ay) / Math.hypot(bx, by) < .7 || step[i] * step[j] < 0) continue;
        seen[j] = 1; queue.push(j);
      }
    }
    const xs = queue.map(i => i % width), ys = queue.map(i => Math.floor(i / width));
    const minX = xs.reduce((a, b) => Math.min(a, b), width), maxX = xs.reduce((a, b) => Math.max(a, b), 0);
    const minY = ys.reduce((a, b) => Math.min(a, b), height), maxY = ys.reduce((a, b) => Math.max(a, b), 0);
    if (queue.length < 10 || Math.hypot(maxX - minX, maxY - minY) < Math.max(18, Math.max(width, height) * .025)) continue;
    const texture = summarizeSeamTexture(queue.filter((_, index) => index % Math.max(1, Math.ceil(queue.length / 64)) === 0).map(i => {
      const [nx, ny] = normals[direction[i]];
      return examineSeamTexture({ width, height, light, valid }, i % width, Math.floor(i / width), nx, ny);
    }));
    const sampled = queue.filter((_, i) => i % Math.max(1, Math.ceil(queue.length / 300)) === 0);
    candidates.push({ texture, id: 0, points: sampled.map(i => [(i % width) / width, Math.floor(i / width) / height]),
      normals: sampled.map(i => normals[direction[i]] as [number, number]),
      exposureDifference: median(queue.map(i => step[i])), confidence: Math.min(.85, median(queue.map(i => quality[i]))),
      bounds: { x: minX / width, y: minY / height, width: (maxX - minX) / width, height: (maxY - minY) / height } });
  }
  return candidates.sort((a, b) => b.confidence - a.confidence).slice(0, 64).map((candidate, id) => ({ ...candidate, id }));
}
