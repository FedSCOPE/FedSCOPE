from __future__ import annotations

import argparse, copy, json
from pathlib import Path
import pandas as pd
import yaml

from .data.opsd import load_household_clients
from .pipeline import run


def _variant(base:dict,name:str,root:Path,**changes):
    cfg=copy.deepcopy(base)
    for dotted,value in changes.items():
        node=cfg; parts=dotted.split("__")
        for k in parts[:-1]: node=node[k]
        node[parts[-1]]=value
    cfg["output"]["directory"]=str(root/name)
    return name,cfg


def main()->None:
    ap=argparse.ArgumentParser(description="Deterministic FedSCOPE paper sensitivity suite")
    ap.add_argument("--config",type=Path,default=Path("configs/paper.yaml")); ap.add_argument("--output-root",type=Path,default=Path("paper_suite_final")); ap.add_argument("--include-focal-rotation",action="store_true"); ap.add_argument("--skip-primary",action="store_true"); args=ap.parse_args()
    base=yaml.safe_load(args.config.read_text(encoding="utf-8")); root=args.output_root; root.mkdir(parents=True,exist_ok=True); variants=[]
    if not args.skip_primary: variants.append(_variant(base,"primary",root))
    variants += [
        _variant(base,"horizon_6h",root,forecast__max_horizon_steps=24,forecast__report_horizons_steps=[1,4,12,24],protocol__control_horizon_steps=24),
        _variant(base,"horizon_12h",root,forecast__max_horizon_steps=48,forecast__report_horizons_steps=[1,4,24,48],protocol__control_horizon_steps=48),
        _variant(base,"coverage_90",root,forecast__control_coverage=0.90),
        _variant(base,"coverage_99",root,forecast__control_coverage=0.99),
        _variant(base,"source_interpolation_sensitivity",root,data__strict_non_interpolated=False),
    ]
    if args.include_focal_rotation:
        clients=load_household_clients(base["data"]["household_csv"],base["scenario"]["essential_feeds"],base["data"]["strict_non_interpolated"])
        pool=[base["scenario"]["focal_client"],*base["scenario"]["peer_clients"]]; measured=sorted({c for c in pool if c in clients and clients[c].measured_critical})
        for focal in measured:
            if focal==base["scenario"]["focal_client"]: continue
            variants.append(_variant(base,f"focal_{focal}",root,scenario__focal_client=focal,scenario__peer_clients=[c for c in pool if c!=focal]))
    rows=[]
    for k,(name,cfg) in enumerate(variants,start=1):
        print(f"\n=== FINAL SUITE {k}/{len(variants)}: {name} ===",flush=True); cp=root/f"{name}.yaml"; cp.write_text(yaml.safe_dump(cfg,sort_keys=False),encoding="utf-8")
        try:
            result=run(cp); rows.append({"variant":name,"status":"ok","evaluated_episodes":result["evaluated_episodes"],"game_active_steps":result["game_active_steps"]})
            for p in result["primary"]: rows.append({"variant":name,"status":"policy","policy":p.get("policy"),**p})
        except Exception as exc:
            rows.append({"variant":name,"status":"failed","error":repr(exc)}); print(f"FAILED {name}: {exc!r}",flush=True)
    pd.DataFrame(rows).to_csv(root/"suite_summary.csv",index=False); (root/"suite_manifest.json").write_text(json.dumps({"design":"deterministic one-factor sensitivity + measured-proxy focal rotation","variants":[n for n,_ in variants],"randomization":"none"},indent=2),encoding="utf-8")
    print(f"\nSuite complete: {root/'suite_summary.csv'}",flush=True)

if __name__=="__main__": main()
