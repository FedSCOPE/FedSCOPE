from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def _tol(values)->float:
    x=np.asarray(values,float); scale=max(1.0,float(np.nanmax(np.abs(x))) if x.size else 1.0); return 512*np.finfo(float).eps*scale


def main()->None:
    ap=argparse.ArgumentParser(description="Scientific audit of FedSCOPE output"); ap.add_argument("results",type=Path,nargs="?",default=Path("results_final")); args=ap.parse_args(); r=args.results
    req=["data_audit.csv","federated_protocol.csv","eb_sharing.csv","forecast_metrics.csv","episodes.csv","game_activity.csv","policy_step_results.csv","episode_metrics.csv","leakage_global.csv","attacker_prequential.csv","policy_effects.csv","run_manifest.json"]
    miss=[x for x in req if not (r/x).exists()]
    if miss: raise SystemExit(f"MISSING OUTPUTS: {miss}")
    prot=pd.read_csv(r/"federated_protocol.csv"); steps=pd.read_csv(r/"policy_step_results.csv"); epm=pd.read_csv(r/"episode_metrics.csv"); game=pd.read_csv(r/"game_activity.csv"); leak=pd.read_csv(r/"leakage_global.csv"); att=pd.read_csv(r/"attacker_prequential.csv"); fc=pd.read_csv(r/"forecast_metrics.csv")
    print("\n=== HARD INVARIANTS ===")
    raw_ok=(~prot.server_receives_raw_rows.astype(bool)).all(); print("server receives no raw rows:","PASS" if raw_ok else "FAIL");
    if not raw_ok: raise SystemExit(2)
    ga=int(game.game_active_steps.sum()); print(f"genuine game-active states: {ga}","PASS" if ga>0 else "FAIL")
    if ga<=0: raise SystemExit("No scarcity competition was observed; strategic claims are not interpretable.")
    real=steps.loc[steps.game_active.astype(bool),"realized_private_horizon"].dropna().to_numpy(float); var_ok=len(real)>1 and np.var(real)>np.finfo(float).eps; print("realized integrated distress non-degenerate:","PASS" if var_ok else "FAIL")
    if not var_ok: raise SystemExit("Realized private target is degenerate on game-active states.")
    p=epm.pivot(index="episode",columns="policy",values="game_unserved_kwh")
    strategic_col = "strategic_executed_opacity"
    if {"minimum_service",strategic_col}.issubset(p.columns):
        diff=p[strategic_col]-p.minimum_service; tol=_tol(p.to_numpy()); ok=bool((diff<=tol).all()); print(f"strategic service preservation vs minimum_service: {'PASS' if ok else 'FAIL'} (tol={tol:.3e})")
        if not ok: print(diff[diff>tol]); raise SystemExit(3)
    print("\n=== GAME ACTIVITY ==="); print(game[["episode","game_active_steps","game_active_fraction","game_supply_kwh","game_focal_deficit_kwh"]].describe().to_string())
    print("\n=== FORECAST: mean MAE across clients/horizons ==="); print(fc.groupby(["target","method"],as_index=False).mae.mean().to_string(index=False))
    print("\n=== LEAKAGE ON GAME-ACTIVE STATES ==="); print(leak.to_string(index=False))
    print("\n=== PREQUENTIAL ATTACKER ==="); print(att.to_string(index=False))
    print("\nAUDIT COMPLETE. Predictive or strategic superiority is deliberately not converted into a hard pass/fail rule.")

if __name__=="__main__": main()
