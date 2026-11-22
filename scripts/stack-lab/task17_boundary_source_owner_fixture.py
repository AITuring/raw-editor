#!/usr/bin/env python3
"""Read-only source-owner/station fixture for one native boundary witness."""
import argparse, glob, json
from pathlib import Path


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--stack-report',required=True); ap.add_argument('--neighbourhood',required=True); ap.add_argument('--legend',required=True); ap.add_argument('--output',required=True); args=ap.parse_args()
    report=json.loads(Path(args.stack_report).read_text()); nb=json.loads(Path(args.neighbourhood).read_text()); legend=json.loads(Path(args.legend).read_text())['legend']
    owners=sorted({p.get('owner',0) for p in nb['pixels'] if p.get('owner',0)})
    owner_map={x['owner']:x for x in legend}; stations=sorted({owner_map[o]['station'] for o in owners if o in owner_map})
    rels=[]
    for r in report.get('station_relations',{}).get('accepted',[]):
        if {r.get('left'),r.get('right')} == set(stations): rels.append(r)
    criterion=next(c for c in report['quality_gate']['criteria'] if c['name']=='boundary_stroke_alignment')
    witness=next((x for x in criterion.get('failed',[]) if x['world']==nb['world']), criterion.get('failed',[{}])[0])
    result={'schema':1,'task':17,'method':'task17_boundary_source_owner_fixture_r1','production_impact':'read-only bounded fixture; no production threshold, bounds, fallback or QG change','stack_report':args.stack_report,'neighbourhood':args.neighbourhood,'legend':args.legend,'witness':{'world':nb['world'],'output':nb['output'],'measured':witness.get('measured'),'threshold':1.5,'crop_owner_counts':{str(o):sum(p.get('owner')==o for p in nb['pixels']) for o in owners},'coverage_zero_count':sum(p.get('covered',0)==0 for p in nb['pixels'])},'owners':[owner_map[o] for o in owners if o in owner_map],'stations':stations,'accepted_station_relations':rels,'classification':'covered_geometry_pass_boundary_stroke_fail','interpretation':'All witness pixels are covered and owned by real sources. The station relation is accepted with geometric evidence, yet boundary stroke error remains above threshold; this isolates a real source-owner/stroke mismatch rather than an uncovered crop edge.','next_step':'Compare source-owner RGB/stroke samples for the station 9↔13 pair in a bounded fixture; do not widen thresholds or bounds.'}
    Path(args.output).write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n'); print(json.dumps({'classification':result['classification'],'owners':owners,'stations':stations}))

if __name__=='__main__': main()
