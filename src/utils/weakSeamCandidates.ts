import { scanNativeSeam, resolveNativeSeamSequences } from './nativeSeamPath';
import type { SeamCandidate } from './seamCandidates';

/** Coherent weak proposals bypass the old per-pixel strong-edge threshold.
 * Each yield is a cancellation/UI scheduling boundary. No pixel is corrected. */
export function* detectWeakSeamCandidates(source: ImageData): Generator<SeamCandidate | null> {
  const scale = Math.min(1, 960 / Math.max(source.width, source.height));
  const width = Math.round(source.width * scale),
    height = Math.round(source.height * scale);
  const data = new Uint8ClampedArray(width * height * 4);
  for (let y = 0; y < height; y++)
    for (let x = 0; x < width; x++) {
      const x0 = Math.floor(x / scale),
        x1 = Math.min(source.width, Math.ceil((x + 1) / scale));
      const y0 = Math.floor(y / scale),
        y1 = Math.min(source.height, Math.ceil((y + 1) / scale));
      const sum = [0, 0, 0, 0];
      let n = 0;
      for (let yy = y0; yy < y1; yy++)
        for (let xx = x0; xx < x1; xx++) {
          const i = (yy * source.width + xx) * 4;
          for (let c = 0; c < 4; c++) sum[c] += source.data[i + c];
          n++;
        }
      for (let c = 0; c < 4; c++) data[(y * width + x) * 4 + c] = sum[c] / n;
    }
  const image = new ImageData(data, width, height),
    found: SeamCandidate[] = [];
  for (const axis of [0, 1]) {
    const size = axis === 0 ? width : height,
      tangent = axis === 0 ? height : width;
    if (tangent < 96 || size < 160) continue;
    const normal: [number, number] = axis === 0 ? [1, 0] : [0, 1];
    for (let center = 64; center < size - 48; center += 80) {
      const positions = Array.from({ length: 8 }, (_, i) => (0.12 + (i * 0.76) / 7) * tangent);
      const scans = positions.map((t) =>
        scanNativeSeam(image, axis === 0 ? center : t, axis === 0 ? t : center, normal, Math.max(32, tangent * 0.095)),
      );
      const results = resolveNativeSeamSequences(scans, undefined, true);
      if (!results.length) yield null;
      for (const result of results) {
        const points: [number, number][] = positions.map((t, i) =>
          axis === 0
            ? [(center + result.offsets[i]) / width, t / height]
            : [t / width, (center + result.offsets[i]) / height],
        );
        const duplicate = found.some(
          (c) => c.normals[0][axis] === 1 && points.every((p, i) => Math.abs(p[axis] - c.points[i][axis]) * size < 6),
        );
        if (duplicate) {
          yield null;
          continue;
        }
        const xs = points.map((p) => p[0]),
          ys = points.map((p) => p[1]);
        const candidate: SeamCandidate = {
          id: found.length,
          points,
          normals: points.map(() => normal),
          exposureDifference: result.delta,
          confidence: Math.min(1, result.validationScore / 10),
          texture: {
            status: 'insufficient',
            correlation: 0,
            energyBalance: 0,
            directionSimilarity: 0,
            shiftedMatch: false,
            residualRatio: 0,
            samples: 0,
          },
          bounds: {
            x: Math.min(...xs),
            y: Math.min(...ys),
            width: Math.max(...xs) - Math.min(...xs),
            height: Math.max(...ys) - Math.min(...ys),
          },
          proposal: 'coherent',
        };
        found.push(candidate);
        yield candidate;
      }
    }
  }
}
