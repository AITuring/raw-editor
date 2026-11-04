export type TextureStatus = 'supported' | 'inconsistent' | 'insufficient';
export interface TextureEvidence {
  status: TextureStatus;
  correlation: number;
  energyBalance: number;
  directionSimilarity: number;
  shiftedMatch: boolean;
  residualRatio: number;
  samples: number;
}
export interface TextureField { width: number; height: number; light: Float32Array; valid: Uint8Array }
const correlation = (a: number[], b: number[]) => {
  const ma = a.reduce((s, v) => s + v, 0) / a.length, mb = b.reduce((s, v) => s + v, 0) / b.length;
  let aa = 0, bb = 0, ab = 0;
  for (let i = 0; i < a.length; i++) { const x = a[i] - ma, y = b[i] - mb; aa += x*x; bb += y*y; ab += x*y; }
  return aa * bb > 1e-12 ? ab / Math.sqrt(aa * bb) : 0;
};

/** Compare independent texture evidence after subtracting each strip's exposure.
 * No luminance-edge thresholds from the candidate generator are used here.
 * Adjacent patches are not identical views of the same scene: lack of correlated
 * texture is inconclusive, unless accompanied by incompatible energy/orientation.
 */
export function examineSeamTexture(field: TextureField, x: number, y: number, nx: number, ny: number): TextureEvidence {
  const { width, height, light, valid } = field;
  const tx = -ny, ty = nx;
  const empty: TextureEvidence = { status: 'insufficient', correlation: 0, energyBalance: 0, directionSimilarity: 0, shiftedMatch: false, residualRatio: 0, samples: 0 };
  const correlations: number[] = [], balances: number[] = [], directions: number[] = [], residuals: number[] = [];
  let shiftedMatch = false;
  for (const radius of [3, 6]) {
    const left: number[] = [], right: number[] = [], leftN: number[] = [], rightN: number[] = [], cross: number[] = [];
    for (let t = -8; t <= 8; t++) {
      const at = (n: number) => {
        const px = x + t * tx + n * nx, py = y + t * ty + n * ny;
        return px >= 0 && py >= 0 && px < width && py < height ? py * width + px : -1;
      };
      const a = at(-radius), b = at(radius), c = at(-radius - 1), d = at(radius + 1), ca = at(-1), cb = at(1);
      if ([a,b,c,d,ca,cb].some(i => i < 0 || !valid[i])) return empty;
      left.push(light[a]); right.push(light[b]); cross.push(light[cb] - light[ca]);
      // Derivatives have the same geometric orientation on both sides.
      leftN.push(light[a] - light[c]); rightN.push(light[d] - light[b]);
    }
    const dl = left.slice(1).map((v, i) => v - left[i]), dr = right.slice(1).map((v, i) => v - right[i]);
    const sum2 = (a: number[]) => a.reduce((s,v)=>s+v*v,0)/a.length;
    const lt = sum2(dl), rt = sum2(dr), ln = sum2(leftN), rn = sum2(rightN);
    const le = lt + ln, re = rt + rn;
    // Texture-free areas cannot establish continuity, even with a perfect step.
    if (Math.max(le,re) < .00015) continue;
    const balance = Math.min(le,re) / Math.max(le,re);
    balances.push(balance);
    // Subtract the robust exposure step, then compare remaining boundary
    // variation with native within-side texture. This also handles stochastic
    // cloth texture, where adjacent strips need not have matching phase.
    const sorted = [...cross].sort((a,b)=>a-b), jump = sorted[Math.floor(sorted.length / 2)];
    residuals.push(Math.sqrt(sum2(cross.map(v=>v-jump)) / Math.max(.00005, ln + rn)));
    const denom = Math.hypot(lt,ln) * Math.hypot(rt,rn);
    directions.push(denom > 1e-12 ? (lt*rt+ln*rn)/denom : 0);
    const corr = correlation(dl,dr);
    correlations.push(corr);
    for (const shift of [-3,-2,2,3]) {
      const l = dl.slice(3,13), r = dr.slice(3+shift,13+shift);
      const c = correlation(l,r);
      if (c > .8 && c > corr + .2) shiftedMatch = true;
    }
  }
  if (!correlations.length) return empty;
  const avg = (v: number[]) => v.reduce((s,x)=>s+x,0)/v.length;
  const corr = avg(correlations), balance = avg(balances), direction = avg(directions), residualRatio = avg(residuals);
  let status: TextureStatus = 'insufficient';
  if (shiftedMatch || balance < .22 || (direction < .65 && corr < .3) || corr < -.35) status = 'inconsistent';
  else if (correlations.length === 2 && ((corr > .65 && balance > .45 && direction > .85) || (balance > .7 && direction > .95 && residualRatio < 1.5))) status = 'supported';
  return { status, correlation: corr, energyBalance: balance, directionSimilarity: direction, shiftedMatch, residualRatio, samples: correlations.length };
}

export function summarizeSeamTexture(evidence: TextureEvidence[]): TextureEvidence {
  const informative = evidence.filter(e => e.samples > 0);
  const avg = (key: 'correlation' | 'energyBalance' | 'directionSimilarity' | 'residualRatio') => informative.reduce((s,e)=>s+e[key],0)/Math.max(1,informative.length);
  const support = evidence.filter(e=>e.status === 'supported').length;
  const inconsistent = evidence.filter(e=>e.status === 'inconsistent').length;
  const status: TextureStatus = evidence.length >= 4 && support / evidence.length >= .65 ? 'supported'
    : evidence.length >= 4 && inconsistent / evidence.length >= .45 ? 'inconsistent' : 'insufficient';
  return { status, correlation: avg('correlation'), energyBalance: avg('energyBalance'), directionSimilarity: avg('directionSimilarity'), shiftedMatch: informative.some(e=>e.shiftedMatch), residualRatio: avg('residualRatio'), samples: informative.length };
}
