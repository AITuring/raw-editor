// Read-only diagnostics. No result from this module enters the production gate.
import fs from 'node:fs';
import crypto from 'node:crypto';

export const identity = [1, 0, 0, 0, 1, 0, 0, 0, 1];
export function multiply(a, b) {
  return Array.from({ length: 9 }, (_, n) => {
    const r = Math.floor(n / 3), c = n % 3;
    return [0, 1, 2].reduce((sum, k) => sum + a[r * 3 + k] * b[k * 3 + c], 0);
  });
}
export function determinant([a, b, c, d, e, f, g, h, i]) {
  return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
}
export function inverse(m) {
  const [a, b, c, d, e, f, g, h, i] = m, det = determinant(m);
  if (!Number.isFinite(det) || Math.abs(det) < 1e-15) throw new Error('Singular frame');
  return [e*i-f*h, c*h-b*i, b*f-c*e, f*g-d*i, a*i-c*g, c*d-a*f,
    d*h-e*g, b*g-a*h, a*e-b*d].map(v => v / det);
}
export function project(m, [x, y]) {
  const w = m[6]*x + m[7]*y + m[8];
  if (!Number.isFinite(w) || Math.abs(w) < 1e-12) throw new Error('Point at infinity');
  return [(m[0]*x + m[1]*y + m[2])/w, (m[3]*x + m[4]*y + m[5])/w];
}
export function localScale(m, [x, y]) {
  const w = m[6]*x + m[7]*y + m[8];
  const value = Math.sqrt(Math.abs(determinant(m) / w**3));
  if (!Number.isFinite(value) || Math.abs(w) < 1e-12) throw new Error('Invalid local scale');
  return value;
}
// Same coordinate map as StackLensCalibration::correct, used only to trace
// measured corrected-grid points back to the original sensor grid. This is
// NOT an uncalibrated RAW remeasurement and cannot add correspondence evidence.
export function correctedToSensor([x,y], { a,b,c, original, corrected }) {
  const dx = x-(corrected[0]-1)/2, dy = y-(corrected[1]-1)/2;
  const radius = Math.hypot(dx,dy)/(Math.min(...original)/2);
  const ratio = radius < 1e-12 ? 1 : a*radius**3+b*radius**2+c*radius+1-a-b-c;
  return [(original[0]-1)/2+dx*ratio,(original[1]-1)/2+dy*ratio];
}
export function distribution(values) {
  const sorted = values.filter(Number.isFinite).sort((a, b) => a - b);
  if (!sorted.length) return { count: 0, min: null, p10: null, median: null, p90: null, max: null };
  const q = p => sorted[Math.round(p * (sorted.length - 1))];
  return { count: sorted.length, min: q(0), p10: q(0.1), median: q(0.5), p90: q(0.9), max: q(1) };
}
const distance = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);
const area = (a, b, c) => ((b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])) / 2;

// These ratios use observed coordinates only: no fitted matrix or reported
// Local_Scale enters the calculation. Long, non-slender triangles limit the
// influence of subpixel localization noise. They are diagnostics, not new
// independent matches, and must never be counted as additional support.
export function observedScale(points, { minEdge = 500, maxEdge = 3000, minAltitude = 200 } = {}) {
  const distances = [], triangles = [], regional = [[], [], [], []];
  if (!points.length) return { distances: distribution([]), triangles: distribution([]), regions: regional.map(distribution) };
  const xs = points.map(p => p.source[0]), ys = points.map(p => p.source[1]);
  const cx = (Math.min(...xs)+Math.max(...xs))/2, cy = (Math.min(...ys)+Math.max(...ys))/2;
  for (let i = 0; i < points.length; i++) for (let j = i+1; j < points.length; j++) {
    const a = points[i], b = points[j], ab = distance(a.source, b.source);
    if (ab < minEdge || ab > maxEdge) continue;
    distances.push(distance(a.target, b.target) / ab);
    for (let k = j+1; k < points.length; k++) {
      const c = points[k], edges = [ab, distance(a.source,c.source), distance(b.source,c.source)];
      if (Math.min(...edges) < minEdge || Math.max(...edges) > maxEdge) continue;
      const from = area(a.source,b.source,c.source), to = area(a.target,b.target,c.target);
      if (2*Math.abs(from)/Math.max(...edges) < minAltitude || from*to <= 0) continue;
      const ratio = Math.sqrt(to/from);
      triangles.push(ratio);
      const x = (a.source[0]+b.source[0]+c.source[0])/3, y = (a.source[1]+b.source[1]+c.source[1])/3;
      regional[(x >= cx ? 1 : 0) + (y >= cy ? 2 : 0)].push(ratio);
    }
  }
  return { distances: distribution(distances), triangles: distribution(triangles), regions: regional.map(distribution) };
}

export function coordinateAudit(report, cache) {
  const source = new Map(cache.sources.map(s => [s.id, s]));
  const poses = cache.global_homographies;
  const referenceInverse = inverse(poses[cache.capture_groups[0].anchor_id]);
  const geometries = new Map();
  for (const c of report.station_relations.candidates) for (const a of [...(c.previous_attempts ?? []), c]) {
    if (a.after_prior_repair) continue;
    geometries.set(a.left, a.left_tile); geometries.set(a.right, a.right_tile);
  }
  return cache.capture_groups.map(group => {
    const anchor = source.get(group.anchor_id), anchorPose = poses[group.anchor_id];
    const layout = multiply(referenceInverse, anchorPose);
    const geometry = geometries.get(group.group_index);
    if (!geometry?.tile_to_world) throw new Error('Missing native tile geometry');
    let crop = multiply(inverse(layout), geometry.tile_to_world);
    crop = crop.map(x => x/crop[8]);
    const cropLinearError = Math.max(...[0,1,3,4,6,7,8].map(i => Math.abs(crop[i]-identity[i])));
    const anchorLocal = multiply(inverse(anchorPose),anchorPose);
    const anchorIdentityError = Math.max(...anchorLocal.map((x,i) => Math.abs(x-identity[i])));
    const measuredAnchor = report.intra_station.find(s => s.station_index === group.group_index)?.anchor_path;
    const memberScales = group.member_ids.map(id => {
      const member = source.get(id), local = multiply(inverse(anchorPose),poses[id]);
      return { path: member.filename, scale_at_source_centre: localScale(local,[member.width/2,member.height/2]) };
    });
    return { station: group.group_index, anchor: anchor.filename, measured_anchor: measuredAnchor,
      anchors_agree: anchor.filename === measuredAnchor, source_dimensions: [anchor.width,anchor.height],
      proxy_scale_factor: anchor.scale_factor, native_tile_dimensions: geometry.dimensions,
      anchor_identity_error: anchorIdentityError, crop, crop_linear_error: cropLinearError, member_scales: memberScales };
  });
}
export function auditReport(report, cache) {
  const frames = coordinateAudit(report,cache);
  const attempts = report.station_relations.candidates.flatMap(c => [...(c.previous_attempts ?? []),c]);
  const rows = attempts.flatMap(attempt => attempt.hypotheses.filter(h => h.selected).map(h => {
    const points = h.correspondence_sample;
    const recomputed = points.map(p => localScale(h.source_to_target_transform,p.source));
    const error = points.map((p,i) => p.local_scale === null ? null : Math.abs(p.local_scale-recomputed[i])).filter(x => x !== null);
    const alternativeAnchors = [];
    if (attempt.seed === 'rendering_prior' && !attempt.after_prior_repair) {
      const sourceGroup = cache.capture_groups[h.canonical_source];
      const targetGroup = cache.capture_groups[h.canonical_target];
      const memberFrame = (group, id) => {
        const local = multiply(inverse(cache.global_homographies[group.anchor_id]),cache.global_homographies[id]);
        return multiply(inverse(local),frames[group.group_index].crop);
      };
      for (const source of sourceGroup.member_ids) for (const target of targetGroup.member_ids) {
        const a = memberFrame(sourceGroup,source), b = memberFrame(targetGroup,target);
        const transformed = points.map(p => ({ source: project(a,p.source), target: project(b,p.target) }));
        alternativeAnchors.push({ source_id: source, target_id: target,
          observed_area_scale: observedScale(transformed).triangles });
      }
    }
    return { pair: [attempt.left,attempt.right], seed: attempt.seed,
      canonical_pair: [h.canonical_source,h.canonical_target], inliers: h.inliers,
      scale_ratio: h.scale_ratio, rejection_reasons: h.rejection_reasons,
      sample_count: points.length, local_scale: distribution(recomputed),
      alternative_anchor_diagnostic_only: alternativeAnchors,
      reported_jacobian_max_error: error.length ? Math.max(...error) : null,
      observed: observedScale(points), holdout_count: h.holdout_count,
      holdout_median_error_px: h.holdout_median_error_px,
      holdout_scale: distribution(h.holdout_transform ? points.map(p => localScale(h.holdout_transform,p.source)) : []) };
  }));
  return { scope: 'Read-only scale diagnosis. Point pairs and triangles are not additional correspondence evidence; no values enter the gate.',
    coordinate_frames: frames, selected_attempts: rows,
    production: { components: report.station_relations.connectivity.components,
      accepted: report.station_relations.accepted.length, quality_gate: report.quality_gate.verdict,
      roi_count: report.quality_gate.roi_count, output_written: report.output.written } };
}
export function fileIdentity(file) {
  return { path: file, sha256: crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex') };
}
