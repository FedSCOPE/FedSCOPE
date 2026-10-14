from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from .data.download import SOURCES
from .data.opsd import load_household_clients, load_german_price, align_price
from .evaluation.leakage import pearson_abs, spearman_abs, distance_correlation, prequential_linear_attacker
from .evaluation.stats import hac_mean_effect, paired_nonparametric
from .forecast.eb_random_effects import (
    STEP, ExpandingStatsCache, EpisodeForecastModel, seasonal_weekly_naive,
)
from .game.controller import EpisodeModels, StepContext, truthful_step, minimum_service_step, strategic_step
from .game.executed_opacity import CausalExecutedDistanceOpacity

METHODS = ("seasonal_naive", "local", "pooled", "federated")
POLICIES = (
    "truthful", "minimum_service", "strategic_local", "strategic_pooled",
    "strategic_federated_no_uq", "strategic_federated", "strategic_oracle",
    "strategic_executed_opacity",
)


def _log(msg: str) -> None:
    print(f"[FedSCOPE] {msg}", flush=True)


def _file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _complete_nonoverlap_episodes(index: pd.DatetimeIndex, steps: int) -> list[pd.DatetimeIndex]:
    idx = index.sort_values()
    if len(idx) == 0:
        return []
    seg_id = idx.to_series().diff().ne(STEP).cumsum().to_numpy()
    out: list[pd.DatetimeIndex] = []
    for gid in np.unique(seg_id):
        seg = idx[seg_id == gid]
        for start in range(0, len(seg) - steps + 1, steps):
            ep = seg[start:start + steps]
            if len(ep) == steps and ep[-1] - ep[0] == (steps - 1) * STEP:
                out.append(ep)
    return out


def _realized_integrated(frame: pd.DataFrame, origin: pd.Timestamp, H: int) -> float:
    idx = pd.date_range(origin, periods=H, freq=STEP, tz=origin.tz)
    if not idx.isin(frame.index).all():
        return np.nan
    g = frame.reindex(idx)
    d = g["critical_deficit_kwh"].to_numpy(float)
    l = g["critical_load_kwh"].to_numpy(float)
    if not np.isfinite(d).all() or not np.isfinite(l).all():
        return np.nan
    den = float(np.sum(np.maximum(l, 0.0)))
    return float(np.sum(np.maximum(d, 0.0)) / den) if den > 0 else 0.0


def _build_caches(target_series: dict[str, dict[str, pd.Series]], cutoffs: list[pd.Timestamp],
                  lags: tuple[int, ...], H: int) -> dict[str, dict[str, ExpandingStatsCache]]:
    caches: dict[str, dict[str, ExpandingStatsCache]] = {}
    for target, mapping in target_series.items():
        caches[target] = {}
        for k, (client, series) in enumerate(mapping.items(), start=1):
            _log(f"building expanding sufficient-statistic cache: {target} {k}/{len(mapping)} {client}")
            caches[target][client] = ExpandingStatsCache(client, series, lags, H, cutoffs)
    return caches


def _fit_episode_models(caches: dict[str, dict[str, ExpandingStatsCache]], cutoff: pd.Timestamp,
                        lags: tuple[int, ...], H: int, focal: str) -> tuple[dict[str, EpisodeForecastModel], list[dict]] | None:
    models: dict[str, EpisodeForecastModel] = {}
    protocol: list[dict] = []
    for target, mapping in caches.items():
        stats = {c: cache.at(cutoff) for c, cache in mapping.items()}
        eligible = {c: st for c, st in stats.items() if st.identifiable()}
        if target in ("critical_deficit", "critical_load") and focal not in eligible:
            return None
        if target == "net_demand" and focal not in eligible:
            return None
        if not eligible:
            return None
        model = EpisodeForecastModel.fit(target, cutoff, lags, H, eligible)
        models[target] = model
        protocol.extend(model.protocol_rows)
    return models, protocol


def _prediction_cache(models: dict[str, EpisodeForecastModel], target_series: dict[str, dict[str, pd.Series]],
                      origins: pd.DatetimeIndex) -> dict[tuple[str, str, str, pd.Timestamp], tuple[np.ndarray, np.ndarray]]:
    cache = {}
    for target, model in models.items():
        for c in model.clients:
            s = target_series[target][c]
            for origin in origins:
                for method in ("local", "pooled", "federated"):
                    cache[(target, method, c, origin)] = model.predict(method, c, s, origin)
    return cache


def _actual_future(series: pd.Series, origin: pd.Timestamp, h: int) -> float:
    return float(series.get(origin + h * STEP, np.nan))


def _append_forecast_errors(store: dict[tuple, list[float]], models: dict[str, EpisodeForecastModel],
                            pred_cache: dict, target_series: dict[str, dict[str, pd.Series]],
                            origins: pd.DatetimeIndex, report_h: list[int], episode: int) -> None:
    for target, model in models.items():
        for c in model.clients:
            s = target_series[target][c]
            for origin in origins:
                seasonal = seasonal_weekly_naive(s, origin, model.H)
                for h in report_h:
                    if h > model.H:
                        continue
                    y = _actual_future(s, origin, h)
                    if not np.isfinite(y):
                        continue
                    preds = {"seasonal_naive": seasonal[h - 1]}
                    sds = {}
                    for method in ("local", "pooled", "federated"):
                        mean, sd = pred_cache[(target, method, c, origin)]
                        preds[method] = mean[h - 1]
                        sds[method] = sd[h - 1]
                    for method, pval in preds.items():
                        if not np.isfinite(pval):
                            continue
                        store[(target, method, c, h, "abs")].append(abs(y - pval))
                        store[(target, method, c, h, "sq")].append((y - pval) ** 2)
                        if method != "seasonal_naive":
                            sdv = sds[method]
                            store[(target, method, c, h, "zerr")].append((y - pval) / sdv if sdv > 0 else np.nan)
                    pf = preds.get("federated", np.nan)
                    if np.isfinite(pf):
                        for base in ("seasonal_naive", "local", "pooled"):
                            pb = preds.get(base, np.nan)
                            if np.isfinite(pb):
                                store[(target, "effect", c, h, base)].append(abs(y - pb) - abs(y - pf))


def _forecast_tables(store: dict[tuple, list[float]], coverage: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    effects = []
    zcrit = float(__import__("scipy.stats", fromlist=["norm"]).norm.ppf(0.5 + 0.5 * coverage))
    keys = sorted({k[:4] for k in store if len(k) == 5 and k[4] == "abs"})
    for target, method, client, h in keys:
        ae = np.asarray(store[(target, method, client, h, "abs")], float)
        se = np.asarray(store[(target, method, client, h, "sq")], float)
        cov = np.nan
        if method != "seasonal_naive" and (target, method, client, h, "zerr") in store:
            z = np.asarray(store[(target, method, client, h, "zerr")], float)
            z = z[np.isfinite(z)]
            cov = float(np.mean(np.abs(z) <= zcrit)) if len(z) else np.nan
        rows.append({"target":target,"method":method,"client":client,"horizon_steps":h,"horizon_hours":h/4,
                     "n":len(ae),"mae":float(np.mean(ae)) if len(ae) else np.nan,
                     "rmse":float(np.sqrt(np.mean(se))) if len(se) else np.nan,
                     "nominal_coverage":coverage if method != "seasonal_naive" else np.nan,
                     "empirical_gaussian_coverage":cov})
    # Paired effects were accumulated only on origins where both forecasts exist.
    effect_keys = sorted(k for k in store if len(k) == 5 and k[1] == "effect")
    for target, _, client, h, base in effect_keys:
        diff = np.asarray(store[(target, "effect", client, h, base)], float)
        if len(diff):
            stat = hac_mean_effect(diff)
            effects.append({"target":target,"client":client,"horizon_steps":h,"baseline":base,**stat})
    return pd.DataFrame(rows), pd.DataFrame(effects)


def _policy_for_origin(context: StepContext, origin: pd.Timestamp, H: int,
                       executed_state: CausalExecutedDistanceOpacity | None = None) -> list[dict]:
    # Compute comparison policies first. The executed-opacity policy reuses the exact
    # current service floor and the causal federated anticipated-distress target.
    base_policies = [p for p in POLICIES if p != "strategic_executed_opacity"]
    out=[]
    by_policy={}
    for policy in base_policies:
        if policy=="truthful": r=truthful_step(context,origin)
        elif policy=="minimum_service": r=minimum_service_step(context,origin)
        elif policy=="strategic_oracle": r=strategic_step(context,origin,"federated",H,use_uncertainty=False,oracle=True)
        else:
            method=policy.replace("strategic_","").replace("_no_uq","")
            r=strategic_step(context,origin,method,H,use_uncertainty=("no_uq" not in policy),oracle=False)
        row={"timestamp":origin,"policy":policy,**r}
        out.append(row); by_policy[policy]=row

    fed=by_policy["strategic_federated"]
    minimum=by_policy["minimum_service"]
    common_private=float(fed.get("private_horizon", np.nan))
    strategic_row=dict(minimum)
    strategic_row["timestamp"]=origin
    strategic_row["policy"]="strategic_executed_opacity"
    strategic_row["private_horizon"]=common_private
    strategic_row["private_current"]=float(fed.get("private_current", np.nan))
    strategic_row["service_floor_current"]=float(minimum.get("signal", 0.0)) if bool(minimum.get("game_active", False)) else 0.0
    if executed_state is not None and bool(minimum.get("game_active", False)):
        dec=executed_state.choose(float(minimum["signal"]), float(by_policy["truthful"]["signal"]), common_private)
        strategic_row["signal"]=float(dec.signal)
        strategic_row["executed_floor_candidate_dcor"]=dec.floor_candidate_dcor
        strategic_row["executed_upper_candidate_dcor"]=dec.upper_candidate_dcor
        strategic_row["executed_truthful_upper"]=bool(dec.used_upper)
        strategic_row["executed_opacity_history_n"]=int(dec.history_size_after)
    else:
        strategic_row["signal"]=0.0 if not bool(minimum.get("game_active", False)) else float(minimum["signal"] )
        strategic_row["executed_floor_candidate_dcor"]=np.nan
        strategic_row["executed_upper_candidate_dcor"]=np.nan
        strategic_row["executed_truthful_upper"]=False
        strategic_row["executed_opacity_history_n"]=executed_state.n if executed_state is not None else 0
    # Every executed-opacity candidate lies on the maximum-service plateau, so the
    # minimum-service allocation is also the allocation of the selected action.
    strategic_row["allocation_source"]="minimum_service_plateau"
    out.append(strategic_row)

    # Apples-to-apples privacy target: all policies are audited against the same
    # causal anticipated-distress value generated at this timestamp.
    for row in out:
        row["common_anticipated_private_horizon"]=common_private
    return out


def _episode_metrics(steps: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    for (ep,policy),g in steps.groupby(["episode","policy"],sort=True):
        ga=g[g.game_active.astype(bool)]
        def dcor(col):
            z=ga[["signal",col]].dropna(); return distance_correlation(z.signal.to_numpy(),z[col].to_numpy()) if len(z)>1 else np.nan
        rows.append({
            "episode":ep,"policy":policy,"steps":len(g),"game_active_steps":len(ga),
            "unserved_kwh":float(g.unserved.sum()),"served_kwh":float(g.served.sum()),"signal_sum":float(g.signal.sum()),
            "game_unserved_kwh":float(ga.unserved.sum()),"game_served_kwh":float(ga.served.sum()),"game_signal_sum":float(ga.signal.sum()),
            "game_private_mean":float(ga.common_anticipated_private_horizon.mean()) if len(ga) else np.nan,
            "game_dcor_internal":dcor("common_anticipated_private_horizon"),"game_dcor_realized":dcor("realized_private_horizon"),
            "signal_total_variation":float(np.abs(np.diff(g.signal.to_numpy(float))).sum()) if len(g)>1 else 0.0,
        })
    return pd.DataFrame(rows)


def _leakage_and_attacker(steps: pd.DataFrame) -> tuple[pd.DataFrame,pd.DataFrame]:
    leak=[]; attacker=[]
    for policy,g0 in steps.groupby("policy",sort=True):
        g=g0[g0.game_active.astype(bool)].sort_values("timestamp")
        for target_name,target in (("common_anticipated_horizon","common_anticipated_private_horizon"),("realized_future_horizon","realized_private_horizon")):
            z=g[["signal",target]].dropna()
            leak.append({"policy":policy,"target":target_name,"n":len(z),
                         "pearson_abs":pearson_abs(z.signal.to_numpy(),z[target].to_numpy()),
                         "spearman_abs":spearman_abs(z.signal.to_numpy(),z[target].to_numpy()),
                         "distance_correlation":distance_correlation(z.signal.to_numpy(),z[target].to_numpy())})
        for target_name,target in (("anticipated","common_anticipated_private_horizon"),("realized","realized_private_horizon")):
            a=g[["timestamp","signal","supply","claimant_count","price_eur_per_mwh",target]].dropna().set_index("timestamp").rename(columns={target:"private"})
            res=prequential_linear_attacker(a) if len(a) else {"n":0,"base_mae":np.nan,"full_mae":np.nan,"mae_reduction":np.nan,"base_r2":np.nan,"full_r2":np.nan,"delta_r2":np.nan}
            attacker.append({"policy":policy,"target":target_name,**res})
    return pd.DataFrame(leak),pd.DataFrame(attacker)


def _policy_effects(metrics: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    comparisons=[("strategic_executed_opacity","truthful"),("strategic_executed_opacity","minimum_service"),("strategic_executed_opacity","strategic_local"),("strategic_executed_opacity","strategic_pooled"),("strategic_executed_opacity","strategic_federated_no_uq"),("strategic_executed_opacity","strategic_federated")]
    for a,b in comparisons:
        A=metrics[metrics.policy==a].set_index("episode"); B=metrics[metrics.policy==b].set_index("episode"); idx=A.index.intersection(B.index)
        for metric in ("game_unserved_kwh","game_signal_sum","game_dcor_internal","game_dcor_realized"):
            if metric not in A or metric not in B: continue
            diff=(B.loc[idx,metric]-A.loc[idx,metric]).to_numpy(float) # positive = A smaller/better for loss/leakage/signal
            h=hac_mean_effect(diff); w=paired_nonparametric(diff); rows.append({"policy_a":a,"policy_b":b,"metric":metric,**h,**{f"np_{k}":v for k,v in w.items()}})
    return pd.DataFrame(rows)


def _sharing_rows(models: dict[str,EpisodeForecastModel], episode:int) -> list[dict]:
    rows=[]
    for target,m in models.items():
        # tau is identical across client coefficient objects; take first.
        first=m.coefficients[m.clients[0]].eb_tau2
        for h,tau in enumerate(first,start=1):
            rows.append({"episode":episode,"target":target,"horizon_steps":h,"tau2_reml":tau,"clients":len(m.clients)})
    return rows


def _figures(steps: pd.DataFrame, metrics: pd.DataFrame, game: pd.DataFrame, out:Path) -> None:
    out.mkdir(parents=True,exist_ok=True)
    if not game.empty and game.game_active_steps.max()>0:
        ep=int(game.sort_values(["game_active_steps","episode"],ascending=[False,True]).iloc[0].episode)
        sub=steps[steps.episode==ep]
        plt.figure(figsize=(12,5))
        for p in ("truthful","minimum_service","strategic_executed_opacity"):
            g=sub[sub.policy==p]; plt.plot(pd.to_datetime(g.timestamp),g.signal,label=p)
        plt.xticks(rotation=30); plt.ylabel("public priority signal"); plt.legend(); plt.tight_layout(); plt.savefig(out/"game_active_signals_example.png",dpi=200); plt.close()
        plt.figure(figsize=(12,5))
        for p in ("truthful","minimum_service","strategic_executed_opacity"):
            g=sub[sub.policy==p]; plt.plot(pd.to_datetime(g.timestamp),g.unserved,label=p)
        plt.xticks(rotation=30); plt.ylabel("unserved measured essential-load proxy [kWh]"); plt.legend(); plt.tight_layout(); plt.savefig(out/"game_active_unserved_example.png",dpi=200); plt.close()
    fm=metrics[(metrics.target=="net_demand") & metrics.method.isin(METHODS)]
    if not fm.empty:
        agg=fm.groupby(["method","horizon_hours"],as_index=False).mae.mean(); plt.figure(figsize=(8,5))
        for m,g in agg.groupby("method"): plt.plot(g.horizon_hours,g.mae,marker="o",label=m)
        plt.xlabel("forecast horizon [h]"); plt.ylabel("mean MAE across clients [kWh]"); plt.legend(); plt.tight_layout(); plt.savefig(out/"forecast_mae_by_horizon.png",dpi=200); plt.close()


def run(config_path: Path) -> dict:
    cfg=yaml.safe_load(Path(config_path).read_text(encoding="utf-8")); out=Path(cfg["output"]["directory"]); out.mkdir(parents=True,exist_ok=True); fig=out/"figures"
    hp=Path(cfg["data"]["household_csv"]); pp=Path(cfg["data"]["price_csv"])
    for p in (hp,pp):
        if not p.exists(): raise FileNotFoundError(f"missing {p}; run downloader")
        exp=SOURCES.get(p.name,{}).get("expected_sha256"); got=_file_hash(p)
        if exp and got!=exp: raise RuntimeError(f"SHA-256 mismatch for {p}: {got} != {exp}")
    clients=load_household_clients(hp,cfg["scenario"]["essential_feeds"],cfg["data"]["strict_non_interpolated"]); price=load_german_price(pp)
    focal=cfg["scenario"]["focal_client"]; pool=[focal,*cfg["scenario"]["peer_clients"]]; selected=[c for c in dict.fromkeys(pool) if c in clients]
    if focal not in selected or not clients[focal].measured_critical: raise ValueError("focal must have measured essential-load proxy")
    claimants=[c for c in selected if clients[c].measured_critical]
    if len(claimants)<2: raise ValueError("need at least two measured essential-load proxy claimants for a strategic competition study")
    frames={c:clients[c].frame for c in selected}; focal_with_price=align_price(frames[focal],price); price15=focal_with_price["price_eur_per_mwh"]
    episodes=_complete_nonoverlap_episodes(focal_with_price.index,int(cfg["protocol"]["episode_steps"])); cutoffs=[ep[0]-STEP for ep in episodes]
    if not episodes: raise ValueError("no complete focal episodes")
    audit=pd.DataFrame([{"client":c,"measured_essential_proxy":clients[c].measured_critical,"rows_strict":len(frames[c]),"start":frames[c].index.min(),"end":frames[c].index.max()} for c in selected]); audit.to_csv(out/"data_audit.csv",index=False)
    H=int(cfg["forecast"]["max_horizon_steps"]); lags=tuple(int(x) for x in cfg["forecast"]["lags_steps"]); coverage=float(cfg["forecast"]["control_coverage"])
    target_series={"net_demand":{c:frames[c]["net_demand_kwh"] for c in selected},"critical_deficit":{c:frames[c]["critical_deficit_kwh"] for c in claimants},"critical_load":{c:frames[c]["critical_load_kwh"] for c in claimants}}
    caches=_build_caches(target_series,cutoffs,lags,H)
    forecast_store=defaultdict(list); protocol_rows=[]; share_rows=[]; step_rows=[]; ep_rows=[]; game_rows=[]; evaluated=0; executed_opacity_state=CausalExecutedDistanceOpacity()
    report_h=[int(x) for x in cfg["forecast"]["report_horizons_steps"]]
    for raw_ep_id,(ep,cutoff) in enumerate(zip(episodes,cutoffs)):
        fit=_fit_episode_models(caches,cutoff,lags,H,focal)
        if fit is None: continue
        models,protocol=fit
        # Need at least focal + one other measured claimant represented in critical models.
        if len(models["critical_deficit"].clients)<2: continue
        evaluated+=1; ep_id=evaluated-1; _log(f"episode {evaluated}: {ep[0]} -> {ep[-1]} (expanding cutoff {cutoff})")
        protocol_rows.extend({"episode":ep_id,**r} for r in protocol); share_rows.extend(_sharing_rows(models,ep_id))
        pred_cache=_prediction_cache(models,target_series,ep); _append_forecast_errors(forecast_store,models,pred_cache,target_series,ep,report_h,ep_id)
        em=EpisodeModels(models=models,series=target_series,coverage=coverage,cache=pred_cache); context=StepContext(frames=frames,focal=focal,selected_clients=selected,measured_claimants=claimants,price=price15,forecasts=em)
        local_rows=[]
        for origin in ep:
            if origin not in price15.index: continue
            rr=_policy_for_origin(context,origin,int(cfg["protocol"]["control_horizon_steps"]),executed_state=executed_opacity_state)
            real=_realized_integrated(frames[focal],origin,int(cfg["protocol"]["control_horizon_steps"]))
            pricev=float(price15.at[origin])
            for r in rr:
                r.update({"episode":ep_id,"raw_episode":raw_ep_id,"price_eur_per_mwh":pricev,"realized_private_horizon":real}); local_rows.append(r)
        if not local_rows: continue
        g=pd.DataFrame(local_rows); step_rows.extend(local_rows)
        truth=g[g.policy=="truthful"]
        game_rows.append({"episode":ep_id,"start":ep[0],"end":ep[-1],"steps":len(truth),"game_active_steps":int(truth.game_active.sum()),"game_active_fraction":float(truth.game_active.mean()),"total_supply_kwh":float(truth.supply.sum()),"total_focal_deficit_kwh":float(truth.focal_deficit.sum()),"game_supply_kwh":float(truth.loc[truth.game_active,"supply"].sum()),"game_focal_deficit_kwh":float(truth.loc[truth.game_active,"focal_deficit"].sum())})
        ep_rows.append({"episode":ep_id,"start":ep[0],"end":ep[-1],"cutoff":cutoff,"participants_net":len(models["net_demand"].clients),"participants_claimants":len(models["critical_deficit"].clients)})
    if evaluated==0: raise ValueError("no identifiable expanding-window episodes")
    pd.DataFrame(protocol_rows).to_csv(out/"federated_protocol.csv",index=False); pd.DataFrame(share_rows).to_csv(out/"eb_sharing.csv",index=False); pd.DataFrame(ep_rows).to_csv(out/"episodes.csv",index=False); game=pd.DataFrame(game_rows); game.to_csv(out/"game_activity.csv",index=False)
    steps=pd.DataFrame(step_rows).sort_values(["episode","timestamp","policy"]); steps.to_csv(out/"policy_step_results.csv",index=False)
    epm=_episode_metrics(steps); epm.to_csv(out/"episode_metrics.csv",index=False)
    fm,fe=_forecast_tables(forecast_store,coverage); fm.to_csv(out/"forecast_metrics.csv",index=False); fe.to_csv(out/"forecast_effects_hac.csv",index=False)
    leak,att=_leakage_and_attacker(steps); leak.to_csv(out/"leakage_global.csv",index=False); att.to_csv(out/"attacker_prequential.csv",index=False)
    effects=_policy_effects(epm); effects.to_csv(out/"policy_effects.csv",index=False); _figures(steps,fm,game,fig)
    primary=epm[epm.policy.isin(["truthful","minimum_service","strategic_executed_opacity"])].groupby("policy").agg(episodes=("episode","count"),game_active_steps=("game_active_steps","sum"),game_unserved_kwh=("game_unserved_kwh","sum"),game_served_kwh=("game_served_kwh","sum"),game_signal_sum=("game_signal_sum","sum"),game_dcor_realized=("game_dcor_realized","mean")).reset_index(); primary.to_csv(out/"primary_summary.csv",index=False)
    manifest={"algorithm":"FedSCOPE: expanding causal empirical-Bayes forecasting + exact game-active proportional fairness + causal executed distance-opacity signaling","config":cfg,"data_hashes":{str(hp):_file_hash(hp),str(pp):_file_hash(pp)},"roles":{"focal":focal,"measured_essential_proxy_claimants":claimants,"all_selected_clients":selected},"episodes":{"candidate":len(episodes),"evaluated":evaluated,"game_active_steps":int(game.game_active_steps.sum())},"randomness":"none in episode construction, client participation, estimator initialization, variance-component estimation, allocation, opacity optimizer, or evaluation","software_versions":{name:importlib.metadata.version(name) for name in ["numpy","pandas","scipy","scikit-learn","matplotlib","PyYAML"]},"scientific_scope":["CoSSMic cumulative meters are converted by endpoint differencing.","Source-marked interpolated endpoints are excluded in strict mode.","Essential feeds are measured essential-load proxies, not certified safety-critical loads.","Islanding is a deterministic counterfactual resilience scenario, not a recorded outage label.","Priority signaling is evaluated primarily on exact game-active states: supply>0, >=2 positive claimants, and supply<total deficit.","Weighted proportional fairness solves max sum_i s_i log(a_i); zero signal is an opt-out and has no equal-priority fallback.","Federated sharing acts only on dimensionless autoregressive lag coefficients; client intercepts remain local.","The random-effects variance is estimated by one-dimensional REML; tau^2=0 yields pooling and large tau^2 approaches local fitting.","Future distress is integrated energy shortfall / integrated essential energy, not a max-over-horizon saturation statistic.","Gaussian predictive intervals are model-based and their empirical coverage is audited; 90/95/99% sensitivity is provided by the suite.","The strategic action is chosen on executed game-active history by cumulative distance correlation; all policies are audited against a common anticipated-distress target.","Federated sufficient-statistic exchange is not differential privacy or a cryptographic guarantee.","No distribution-network topology exists in CoSSMic; power-flow security is outside the dataset-backed claim."]}
    (out/"run_manifest.json").write_text(json.dumps(manifest,indent=2,default=str),encoding="utf-8")
    _log(f"complete: {out.resolve()}"); print(primary.to_string(index=False),flush=True)
    return {"output":str(out),"evaluated_episodes":evaluated,"game_active_steps":int(game.game_active_steps.sum()),"primary":primary.to_dict("records")}


def main() -> None:
    ap=argparse.ArgumentParser(description="Run the FedSCOPE expanding real-data experiment"); ap.add_argument("--config",type=Path,default=Path("configs/paper.yaml")); args=ap.parse_args(); run(args.config)


if __name__=="__main__": main()
