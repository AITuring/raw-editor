#!/usr/bin/env python3
"""PF.8 test-only permutation/null distribution on saved Langyuan proxies."""
from __future__ import annotations
import argparse, hashlib, json, math, random, statistics, subprocess
from pathlib import Path
ROOT=Path('/Users/dp/Desktop/raw-editor')
ALIGNMENT=ROOT/'output/acceptance/task-5/langyuan-baseline-r1/source-alignment.json'
THUMBS=ROOT/'work/analysis/langyuan_thumbs'
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def proxy(path):
 r=subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-threads','1','-i',str(path),'-vf','scale=400:267','-frames:v','1','-f','rawvideo','-pix_fmt','gray','-'],check=True,capture_output=True)
 if len(r.stdout)!=400*267:raise ValueError(f'unexpected proxy size {path}: {len(r.stdout)}')
 return 400,267,r.stdout
def raw_gray(path):
 r=subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-threads','1','-i',str(path),'-f','rawvideo','-pix_fmt','gray','-'],check=True,capture_output=True)
 return r.stdout
def transform(h,x,y):
 z=h[6]*x+h[7]*y+h[8]
 if not math.isfinite(z) or abs(z)<1e-12:return None
 qx=(h[0]*x+h[1]*y+h[2])/z; qy=(h[3]*x+h[4]*y+h[5])/z
 return (qx,qy) if math.isfinite(qx) and math.isfinite(qy) else None
def ncc(a,b):
 if len(a)<64 or len(a)!=len(b):return None
 ma,mb=statistics.fmean(a),statistics.fmean(b);da=[v-ma for v in a];db=[v-mb for v in b];va,vb=sum(v*v for v in da),sum(v*v for v in db)
 if va<=1e-9 or vb<=1e-9:return None
 return sum(x*y for x,y in zip(da,db))/math.sqrt(va*vb)
def score(source,target,h,sd,td,native_source=(8258,5507),native_target=(8258,5507)):
 sw,sh=sd;tw,th=td;left=[];right=[]
 for row in range(2,27):
  for col in range(2,40):
   sx=(col+.5)*sw/40.;sy=(row+.5)*sh/27.;mapped=transform(h,sx*native_source[0]/sw,sy*native_source[1]/sh)
   if mapped is None:continue
   ix,iy=round(mapped[0]*tw/native_target[0]),round(mapped[1]*th/native_target[1])
   if 1<=ix<tw-1 and 1<=iy<th-1:
    idx=int(sy)*sw+int(sx);ti=iy*tw+ix;left.extend([float(source[idx]),float(source[idx+1])-float(source[idx-1])]);right.extend([float(target[ti]),float(target[ti+1])-float(target[ti-1])])
 v=ncc(left,right);return None if v is None else {'ncc':v,'samples':len(left)//2}
def quantiles(values):
 o=sorted(values)
 if not o:return {'count':0}
 at=lambda p:o[min(len(o)-1,int(math.floor(p*len(o))))]
 return {'count':len(o),'p50':at(.5),'p95':at(.95),'p99':at(.99),'max':o[-1]}
def main():
 p=argparse.ArgumentParser();p.add_argument('--permutations',type=int,default=1000);p.add_argument('--seed',type=int,default=0x7F8_20261118);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 if a.permutations<100:raise SystemExit('at least 100 permutations are required')
 alignment=json.loads(ALIGNMENT.read_text());sources={x['id']:x for x in alignment['sources']};groups={}
 for gi,g in enumerate(alignment['capture_groups']):
  for sid in g['member_ids']:groups[sid]=gi
 files={sid:THUMBS/(Path(x['filename']).name+'.png') for sid,x in sources.items()};missing=[str(x) for x in files.values() if not x.is_file()]
 if missing:raise SystemExit(f'missing proxies: {missing[:3]}')
 images={sid:proxy(path) for sid,path in files.items()};accepted=[m for m in alignment['matches'] if m['inliers']>=15 and m['source_id'] in groups and m['target_id'] in groups];adjacent=[m for m in accepted if abs(groups[m['source_id']]-groups[m['target_id']])<=1];candidates=[m for m in adjacent if groups[m['source_id']]!=groups[m['target_id']]]
 if not candidates:raise SystemExit('no accepted cross-group edges')
 adj=[]
 for e in candidates:
  r=score(images[e['source_id']][2],images[e['target_id']][2],e['direct_homography'],images[e['source_id']][:2],images[e['target_id']][:2])
  if r:adj.append({'source':e['source_id'],'target':e['target_id'],**r})
 rng=random.Random(a.seed);perms=[];ids=sorted(sources)
 for index in range(a.permutations):
  e=candidates[rng.randrange(len(candidates))];sid=e['source_id'];pool=[tid for tid in ids if abs(groups[tid]-groups[sid])>1 and tid!=sid];tid=pool[rng.randrange(len(pool))];r=score(images[sid][2],images[tid][2],e['direct_homography'],images[sid][:2],images[tid][:2])
  if r:perms.append({'index':index,'source':sid,'target':tid,'source_group':groups[sid],'target_group':groups[tid],**r})
 nv=[x['ncc'] for x in perms];av=[x['ncc'] for x in adj];threshold=quantiles(nv).get('p99')
 joint_dir=ROOT/'output/acceptance/langyuan-v44-joint-field/candidate-repeat/langyuan-10/planes';joint=json.loads((joint_dir/'joint-production-input.json').read_text());stations=[]
 for i,(width,height) in enumerate(joint['dimensions']):stations.append((width,height,raw_gray(joint_dir/f'station-{i:03d}.gray.png')))
 edge02=next(r for r in joint['records'] if r['left']==0 and r['right']==2);raw_h=edge02['homography'];edge02_h=[raw_h[0],raw_h[3],raw_h[6],raw_h[1],raw_h[4],raw_h[7],raw_h[2],raw_h[5],raw_h[8]];edge02_score=score(stations[0][2],stations[2][2],edge02_h,stations[0][:2],stations[2][:2],joint['dimensions'][0],joint['dimensions'][2])
 report={'schema':1,'method':'saved_proxy_global_pose_permutation_ncc_v1','scope':'Langyuan84 baseline proxies; non-adjacent source replacement with accepted-edge geometry','production_impact':'test_only_no_stack_report_no_threshold_no_pose_write','seed':a.seed,'permutations_requested':a.permutations,'permutations_usable':len(perms),'accepted_cross_group_edges':len(candidates),'adjacent_real_scores':quantiles(av),'non_adjacent_null_scores':quantiles(nv),'null_p99_threshold':threshold,'real_scores_above_null_p99':sum(v>threshold for v in av) if threshold is not None else None,'langyuan_0_2_edge':{'record_inliers':edge02['inliers'],'record_error':edge02['error'],'record_hull_support':edge02.get('diagnostic',{}).get('hull_support'),'proxy_ncc':edge02_score,'homography_layout':'transposed_from_nalgebra_flattened_record','planes':str(joint_dir)},'sources':{'alignment':str(ALIGNMENT),'alignment_sha256':sha(ALIGNMENT),'proxy_directory':str(THUMBS),'proxy_sha256':{str(i):sha(path) for i,path in files.items()}},'selection':{'group_distance_for_null':'>1','edge_inlier_minimum':15,'grid':'38x25 interior points, source-to-target direct homography from saved accepted edge','score':'interleaved intensity and horizontal-gradient NCC','proxy_dimensions':[400,267]},'samples':{'adjacent':adj,'permutations':perms}}
 a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n');print(json.dumps({k:report[k] for k in ('accepted_cross_group_edges','permutations_requested','permutations_usable','adjacent_real_scores','non_adjacent_null_scores','null_p99_threshold','real_scores_above_null_p99','langyuan_0_2_edge')},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
