#!/usr/bin/env python3
"""Test-only audit of saved Qixu owner/coverage maps; never rewrites pixels."""
import argparse,json,subprocess,collections
from pathlib import Path

def raw(path,pix):
 return subprocess.check_output(['ffmpeg','-hide_banner','-loglevel','error','-i',str(path),'-f','rawvideo','-pix_fmt',pix,'-'])
def main():
 p=argparse.ArgumentParser();p.add_argument('--diagnostics',type=Path,required=True);p.add_argument('--report',type=Path,required=True);p.add_argument('--analysis',type=Path,required=True);a=p.parse_args()
 d=json.loads((a.diagnostics/'metadata.json').read_text());owner=raw(a.diagnostics/'owner.png','gray');coverage=raw(a.diagnostics/'coverage.png','gray');ow,oh=d['analysis_width'],d['analysis_height'];origin=(-4581.0,1.0);scale=d['analysis_scale'];report=json.loads(json.loads(a.analysis.read_text())['report'] and Path(json.loads(a.analysis.read_text())['report']).read_text());failures=[]
 for criterion in report['quality_gate']['criteria']:
  for item in criterion.get('failed',[]):
   if 'world' not in item:continue
   wx,wy=item['world']['x'],item['world']['y'];ax=round((wx-origin[0])/scale);ay=round((wy-origin[1])/scale);vals=[];covered=0
   for dy in range(-1,2):
    for dx in range(-1,2):
     xx=max(0,min(ow-1,ax+dx));yy=max(0,min(oh-1,ay+dy));vals.append(owner[yy*ow+xx]);covered+=coverage[yy*ow+xx]>0
   counts=collections.Counter(vals);center=owner[max(0,min(oh-1,ay))*ow+max(0,min(ow-1,ax))];alternates=[v for v,n in counts.items() if v not in (0,center) and n>=2]
   failures.append({'criterion':criterion['name'],'world':[wx,wy],'analysis':[ax,ay],'center_owner':center,'owner_counts':dict(counts),'covered_neighbors':covered,'alternate_candidates':alternates,'fallback_candidate':bool(alternates) and covered==9})
 scale_fail=[x for x in failures if x['criterion'] in ('local_scale_median','local_scale_pixel_ratio')];boundary=[x for x in failures if x['criterion']=='boundary_stroke_alignment'];owner_fail=[x for x in failures if x['criterion']=='owner_sharpness_coverage']
 out={'schema':1,'method':'qixu_saved_owner_3x3_fallback_audit_v1','production_impact':'test_only_no_pixel_write_no_owner_change','diagnostics':str(a.diagnostics),'metadata':d,'origin_world':origin,'failures':failures,'summary':{'scale_failure_points':len(scale_fail),'scale_points_with_alternate_candidate':sum(x['fallback_candidate'] for x in scale_fail),'boundary_failure_points':len(boundary),'boundary_points_with_alternate_candidate':sum(x['fallback_candidate'] for x in boundary),'owner_coverage_failure_points':len(owner_fail),'owner_coverage_points_with_covered_neighbors':sum(x['covered_neighbors']>0 for x in owner_fail),'decision':'do_not_apply_fallback_automatically: scale failures have no alternate 3x3 owner; boundary alternates are geometry candidates only and require independent resampling/QG evidence','fallback_scope':'same 3x3 owner stencil, Coverage required, owner=0 excluded, no search expansion'}}
 a.report.parent.mkdir(parents=True,exist_ok=True);a.report.write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n');print(json.dumps(out['summary'],ensure_ascii=False,indent=2))
if __name__=='__main__':main()
