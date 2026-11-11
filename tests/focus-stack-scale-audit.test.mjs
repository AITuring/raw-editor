import test from 'node:test';
import assert from 'node:assert/strict';
import { identity, multiply, inverse, project, localScale, observedScale, correctedToSensor } from './support/station-scale-audit.mjs';

const pointsFor = matrix => Array.from({ length: 36 }, (_, i) => {
  const source = [300+(i%6)*700, 400+Math.floor(i/6)*650];
  return { source, target: project(matrix,source) };
});
test('observed distances distinguish unit scale and the rejected real-data scale range without a model', () => {
  for (const scale of [0.9,1,1.12]) {
    const points = pointsFor([scale,0,130,0,scale,-60,0,0,1]);
    const result = observedScale(points);
    assert.ok(result.triangles.count > 24);
    assert.ok(Math.abs(result.triangles.median-scale) < 1e-12);
    assert.ok(Math.abs(result.distances.median-scale) < 1e-12);
    assert.ok(result.regions.every(r => r.count > 0 && Math.abs(r.median-scale) < 1e-12));
  }
});
test('translation, rotation and independent crop origins preserve area scale; shear is not mistaken for isotropic zoom', () => {
  const angle = 0.31, c = Math.cos(angle), s = Math.sin(angle);
  const matrix = multiply([c,-s,230,s,c,90,0,0,1], [1.2,0.2,0,0,1/1.2,0,0,0,1]);
  const points = pointsFor(matrix).map(p => ({ source: [p.source[0]-113,p.source[1]+301],target: [p.target[0]+201,p.target[1]-71] }));
  assert.ok(Math.abs(observedScale(points).triangles.median-1) < 1e-12);
});
test('Jacobian scale agrees with finite differences and is invariant to homogeneous matrix normalization', () => {
  const matrix = [1.1,0.02,250,-0.01,1.09,-110,0.000002,-0.000004,1];
  for (const point of [[100,200],[3000,2400],[7000,5000]]) {
    const epsilon = 0.001, a = project(matrix,point);
    const b = project(matrix,[point[0]+epsilon,point[1]]), c = project(matrix,[point[0],point[1]+epsilon]);
    const numeric = Math.sqrt(Math.abs(((b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0]))/epsilon**2));
    assert.ok(Math.abs(localScale(matrix,point)-numeric) < 1e-6);
    assert.ok(Math.abs(localScale(matrix.map(v=>v*17),point)-numeric) < 1e-6);
  }
});
test('anchor inversion cancels projective placement and isolates a pure crop translation', () => {
  const anchor = [0.9,-0.02,220,0.01,0.91,3100,0.000002,-0.000004,1];
  const crop = [1,0,-31,0,1,-47,0,0,1];
  const recovered = multiply(inverse(anchor),multiply(anchor,crop));
  recovered.forEach((v,i)=>assert.ok(Math.abs(v-crop[i]) < 1e-10));
  multiply(inverse(anchor),anchor).forEach((v,i)=>assert.ok(Math.abs(v-identity[i]) < 1e-10));
});
test('missing, collinear or reflected observations cannot manufacture area evidence', () => {
  for (const points of [[], Array.from({length:10},(_,i)=>({source:[i*700,0],target:[i*700,0]})), pointsFor([-1,0,6000,0,1,0,0,0,1])]) {
    const result = observedScale(points);
    assert.equal(result.triangles.count,0);
    assert.equal(result.triangles.median,null);
  }
  assert.throws(()=>inverse(Array(9).fill(0)),/Singular/);
});
test('common proxy scaling preserves measured scale while unequal sampling factors must be accounted for', () => {
  const original = pointsFor([0.91,0,100,0,0.91,-80,0,0,1]);
  const originalScale = observedScale(original).triangles.median;
  const scaled = (a,b) => original.map(p => ({source:p.source.map(v=>v*a),target:p.target.map(v=>v*b)}));
  const half = observedScale(scaled(0.5,0.5),{minEdge:250,maxEdge:1500,minAltitude:100});
  assert.ok(Math.abs(half.triangles.median-originalScale)<1e-12);
  const unequal = observedScale(scaled(1,0.5));
  assert.ok(Math.abs(unequal.triangles.median-originalScale/2)<1e-12);
  assert.notEqual(unequal.triangles.median,originalScale);
});
test('optical backprojection uses the original short-side radius and does not scale by canvas size', () => {
  const calibration = {a:0,b:0,c:0,original:[8000,5000],corrected:[8004,5006]};
  assert.deepEqual(correctedToSensor([4001.5,2502.5],calibration),[3999.5,2499.5]);
  assert.deepEqual(correctedToSensor([5001.5,3502.5],calibration),[4999.5,3499.5]);
  assert.deepEqual(correctedToSensor([6501.5,2502.5],{...calibration,a:0.02,b:-0.01,c:0.03}),[6499.5,2499.5]);
});
