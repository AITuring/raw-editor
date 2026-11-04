import assert from 'node:assert/strict';
import { build } from 'esbuild';
globalThis.ImageData = class { constructor(data,width,height) { Object.assign(this,{data,width,height}); } };
const bundle=await build({entryPoints:['src/utils/seamCandidates.ts'],bundle:true,format:'esm',write:false});
const {detectSeamCandidates}=await import('data:text/javascript;base64,'+Buffer.from(bundle.outputFiles[0].contents).toString('base64'));
function make(pixel){const w=240,h=180,data=new Uint8ClampedArray(w*h*4);for(let y=0;y<h;y++)for(let x=0;x<w;x++){const value=pixel(x,y);data.set(Array.isArray(value)?value:[value,value,value,255],(y*w+x)*4);}return new ImageData(data,w,h);}
assert.equal(detectSeamCandidates(make(()=>120)).length,0);
assert.equal(detectSeamCandidates(make(x=>80+x*.4)).length,0,'smooth gradients are not seams');
assert.equal(detectSeamCandidates(make(x=>Math.abs(x-100)<2?60:120)).length,0,'thin scratches are not exposure steps');
assert.equal(detectSeamCandidates(make(x=>x<100?[140,50,25,255]:[25,50,140,255])).length,0,'chromatic object edges must be rejected');
assert.equal(detectSeamCandidates(make(x=>x<100?[90,90,90,0]:[120,120,120,255])).length,0);
for(const pixel of [x=>x<100?100:130,(x,y)=>y<90?100:130,(x,y)=>x<80+y*.3?100:130,(x,y)=>x<110+15*Math.sin(y/45)?100:130]){
 const candidates=detectSeamCandidates(make(pixel));
 assert.ok(candidates.length>0,'detect straight, diagonal and curved exposure steps');
 assert.ok(candidates.some(c=>Math.abs(Math.abs(c.exposureDifference)-Math.log2(1.3))<.02),'estimate paired luminance jump');
 assert.ok(candidates.every(c=>c.points.every(p=>p.every(v=>v>=0&&v<=1))));
}
// A grayscale material edge is observationally ambiguous: never return a certainty of 1.
assert.ok(detectSeamCandidates(make(x=>x<100?100:130)).every(c=>c.confidence<1));
console.log('candidate detector: steps, curves, ramps, scratches, chroma and alpha passed');
