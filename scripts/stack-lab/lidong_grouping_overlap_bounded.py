#!/usr/bin/env python3
"""Test-only audit for LiDong geometry-disconnected components.

Reads an existing Stack_Report and evaluates whether already-measured rejected
station/tile relations can provide a bounded bridge between disconnected
components without changing production gates. It never mutates source data or
re-runs the compositor.
"""
from __future__ import annotations
import argparse, json, math, re
from collections import Counter, defaultdict
from pathlib import Path

GROUP_RE = re.compile(r"focus-group-(\d+)-")

def group_index(member: str) -> int:
    m = GROUP_RE.search(member)
    if not m:
        raise ValueError(f"unparseable virtual group: {member}")
    # Relation graph indexes are zero-based while virtual group labels are 1-based.
    return int(m.group(1)) - 1

def hypotheses(entry):
    out=[]
    for attempt in [entry, *entry.get("previous_attempts", [])]:
        for h in attempt.get("hypotheses", []):
            row=dict(h)
            row["_attempt"] = attempt
            out.append(row)
    return out

def best_bridge(entries, left_nodes, right_nodes):
    rows=[]
    for e in entries:
        l,r=e.get("left"),e.get("right")
        if {l,r} != {left_nodes[0], right_nodes[0]}:
            pass
        if l not in left_nodes or r not in right_nodes:
            continue
        if (l in left_nodes) == (r in left_nodes):
            continue
        hs=hypotheses(e)
        for h in hs:
            rows.append((h,e))
    if not rows:
        return None
    # Prefer the strongest measured bridge, then the lowest residual. This is
    # diagnostic ranking only; no gate is changed and no candidate is accepted.
    def key(pair):
        h,e=pair
        return (int(h.get("inliers") or 0), -float(h.get("median_error_px") or 1e9), float(h.get("spatial_support") or 0.0))
    h,e=max(rows,key=key)
    a=h.get("_attempt",e)
    return {
        "left": e.get("left"), "right": e.get("right"),
        "model": h.get("model"), "selected": h.get("selected",False),
        "inliers": h.get("inliers"), "median_error_px": h.get("median_error_px"),
        "scale_ratio": h.get("scale_ratio"), "spatial_support": h.get("spatial_support"),
        "edge_strength_ratio": h.get("edge_strength_ratio"),
        "holdout_median_error_px": (h.get("scale_evidence") or {}).get("holdout_median_error_px"),
        "holdout_spatial_support": (h.get("scale_evidence") or {}).get("holdout_spatial_support"),
        "scale_evidence_passes": (h.get("scale_evidence") or {}).get("passes"),
        "rejection_reasons": h.get("rejection_reasons",[]),
        "failure_stage": a.get("failure_stage"),
        "bidirectional_match_count": a.get("bidirectional_match_count"),
        "measurable_patch_count": a.get("measurable_patch_count"),
        "predicted_overlap_area_px": a.get("predicted_overlap_area_px"),
        "hull_support": a.get("hull_support"),
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--stack-report", required=True)
    ap.add_argument("--out", required=True)
    args=ap.parse_args()
    report=json.loads(Path(args.stack_report).read_text())
    disconnected=[]
    for e in report.get("degradation",{}).get("entries",[]):
        if e.get("reason")=="geometry_disconnected":
            disconnected=e.get("detail",{}).get("components",[])
            break
    if len(disconnected)!=3:
        raise SystemExit(f"expected 3 disconnected components, found {len(disconnected)}")
    # Preserve report component order, and translate virtual group labels to relation nodes.
    components=[]
    for i,c in enumerate(disconnected):
        nodes=sorted(group_index(m) for m in c.get("members",[]))
        components.append({"component":i,"member_count":c.get("member_count"),"virtual_members":c.get("members",[]),"relation_nodes":nodes})
    accepted=report.get("station_relations",{}).get("accepted",[])
    candidates=report.get("station_relations",{}).get("candidates",[])
    comp_by_node={n:c["component"] for c in components for n in c["relation_nodes"]}
    accepted_cross=[]
    for e in accepted:
        l,r=e.get("left"),e.get("right")
        if l in comp_by_node and r in comp_by_node and comp_by_node[l]!=comp_by_node[r]:
            accepted_cross.append({"left":l,"right":r,"components":[comp_by_node[l],comp_by_node[r]],"inliers":e.get("inliers"),"scale_ratio":e.get("scale_ratio"),"spatial_support":e.get("spatial_support")})
    bridge_summary=[]
    for i in range(len(components)):
        for j in range(i+1,len(components)):
            lnodes=components[i]["relation_nodes"]; rnodes=components[j]["relation_nodes"]
            relevant=[]
            for e in candidates:
                l,r=e.get("left"),e.get("right")
                if (l in lnodes and r in rnodes) or (l in rnodes and r in lnodes): relevant.append(e)
            bridge=best_bridge(relevant,lnodes,rnodes) if relevant else None
            bridge_summary.append({"components":[i,j],"candidate_count":len(relevant),"bridge":bridge,"gate_passing_bridge_count":sum(1 for e in relevant for h in hypotheses(e) if not h.get("rejection_reasons") and (h.get("scale_evidence") or {}).get("passes") is not False)})
    # Explicit bounded graph A/B: force only the strongest measured bridge(s)
    # as a synthetic connectivity experiment. This is not an acceptance path;
    # the injected edges retain their existing rejection reasons.
    def graph_components(extra_edges):
        parent={c["component"]:c["component"] for c in components}
        def find(x):
            while parent[x]!=x:
                parent[x]=parent[parent[x]]; x=parent[x]
            return x
        def union(a,b):
            a,b=find(a),find(b)
            if a!=b: parent[b]=a
        for x in extra_edges: union(*x)
        return len({find(x) for x in parent})
    strong_edges=[]
    for x in bridge_summary:
        b=x.get("bridge")
        if b is not None:
            strong_edges.append(tuple(x["components"]))
    variants=[
      {"id":"baseline","injected_edges":[],"remaining_component_count":len(components),"gate_compatible":True},
      {"id":"force_best_c0_c1","injected_edges":[list(strong_edges[0])] if len(strong_edges)>0 else [],"remaining_component_count":graph_components(strong_edges[:1]),"gate_compatible":False},
      {"id":"force_best_c1_c2","injected_edges":[list(strong_edges[-1])] if len(strong_edges)>1 else [],"remaining_component_count":graph_components(strong_edges[-1:]),"gate_compatible":False},
      {"id":"force_both_cross_bridges","injected_edges":[list(x) for x in strong_edges],"remaining_component_count":graph_components(strong_edges),"gate_compatible":False},
    ]
    result={
      "schema":1,"analysis_type":"task17_lidong_grouping_overlap_bounded",
      "production_impact":"test-only report; no source, threshold, bounds, fallback, or compositor changes",
      "input_stack_report":args.stack_report,
      "selected_path":report.get("selected_path"),
      "component_count":len(components),"components":components,
      "accepted_cross_component_edges":accepted_cross,
      "bridge_summary":bridge_summary,
      "bounded_variants":variants,
      "conclusion":{
        "accepted_cross_component_edge_count":len(accepted_cross),
        "gate_passing_bridge_count":sum(x["gate_passing_bridge_count"] for x in bridge_summary),
        "bounded_grouping_result":"three disconnected components remain; no measured cross-component edge passes the existing relation gates",
        "next_experiment":"only test explicit source regrouping/overlap fixtures per component; do not relax station scale or edge-strength gates"
      }
    }
    Path(args.out).write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(result["conclusion"],ensure_ascii=False,indent=2))

if __name__=="__main__": main()
