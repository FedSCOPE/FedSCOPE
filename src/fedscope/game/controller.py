from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.stats import norm

from fedscope.forecast.eb_random_effects import EpisodeForecastModel, STEP
from .allocation import game_active, truthful_viability_signal, weighted_proportional_fair_allocation
from .opacity import minimum_service_preserving_signal, plan_minimum_distortion_opacity


@dataclass(frozen=True)
class EpisodeModels:
    models: dict[str, EpisodeForecastModel]  # net_demand, critical_deficit, critical_load
    series: dict[str, dict[str, pd.Series]]
    coverage: float
    cache: dict[tuple[str, str, str, pd.Timestamp], tuple[np.ndarray, np.ndarray]] | None = None


@dataclass(frozen=True)
class StepContext:
    frames: dict[str, pd.DataFrame]
    focal: str
    selected_clients: list[str]
    measured_claimants: list[str]
    price: pd.Series
    forecasts: EpisodeModels


def _row(frame: pd.DataFrame, origin: pd.Timestamp) -> pd.Series | None:
    if origin not in frame.index:
        return None
    r = frame.loc[origin]
    if isinstance(r, pd.DataFrame):
        r = r.iloc[-1]
    return r


def _actual_market(context: StepContext, origin: pd.Timestamp) -> dict:
    claimants: list[str] = []
    deficits: list[float] = []
    loads: list[float] = []
    for c in context.measured_claimants:
        r = _row(context.frames[c], origin)
        if r is None:
            continue
        d = max(float(r["critical_deficit_kwh"]), 0.0)
        l = max(float(r["critical_load_kwh"]), 0.0)
        if d > 0.0:
            claimants.append(c); deficits.append(d); loads.append(l)
    supply = 0.0
    suppliers: list[str] = []
    for c in context.selected_clients:
        if c == context.focal:
            continue
        r = _row(context.frames[c], origin)
        if r is None:
            continue
        surplus = max(-float(r["net_demand_kwh"]), 0.0)
        if surplus > 0.0:
            suppliers.append(c); supply += surplus
    d = np.asarray(deficits, dtype=float)
    l = np.asarray(loads, dtype=float)
    signals = truthful_viability_signal(d, l)
    return {
        "claimants": claimants,
        "deficits": d,
        "loads": l,
        "truthful_signals": signals,
        "supply": float(supply),
        "suppliers": suppliers,
        "game_active": game_active(d, supply),
    }


def _focal_position(market: dict, focal: str) -> int | None:
    try:
        return market["claimants"].index(focal)
    except ValueError:
        return None


def _allocate_actual(market: dict, focal: str, focal_signal: float) -> tuple[float, float, float]:
    pos = _focal_position(market, focal)
    if pos is None:
        return 0.0, 0.0, 0.0
    sig = market["truthful_signals"].copy()
    if market["game_active"]:
        sig[pos] = max(float(focal_signal), 0.0)
    else:
        sig[pos] = 0.0
    allocation = weighted_proportional_fair_allocation(market["deficits"], sig, market["supply"])
    d = float(market["deficits"][pos])
    served = float(allocation[pos])
    return d, served, max(d - served, 0.0)


def truthful_step(context: StepContext, origin: pd.Timestamp) -> dict:
    market = _actual_market(context, origin)
    pos = _focal_position(market, context.focal)
    if pos is None:
        private = 0.0; signal = 0.0
    else:
        private = float(market["truthful_signals"][pos])
        signal = private if market["game_active"] else 0.0
    d, served, unserved = _allocate_actual(market, context.focal, signal)
    return _base_result(market, d, signal, served, unserved, private, private)


def minimum_service_step(context: StepContext, origin: pd.Timestamp) -> dict:
    market = _actual_market(context, origin)
    pos = _focal_position(market, context.focal)
    if pos is None:
        private = 0.0; signal = 0.0
    else:
        private = float(market["truthful_signals"][pos])
        if market["game_active"]:
            peer_mask = np.arange(len(market["deficits"])) != pos
            signal, _ = minimum_service_preserving_signal(
                market["deficits"][pos], market["deficits"][peer_mask],
                market["truthful_signals"][peer_mask], market["supply"]
            )
        else:
            signal = 0.0
    d, served, unserved = _allocate_actual(market, context.focal, signal)
    return _base_result(market, d, signal, served, unserved, private, private)


def _base_result(market: dict, deficit: float, signal: float, served: float, unserved: float,
                 private_current: float, private_horizon: float) -> dict:
    return {
        "signal": float(signal), "served": float(served), "unserved": float(unserved),
        "focal_deficit": float(deficit), "private_current": float(private_current),
        "private_horizon": float(private_horizon), "supply": float(market["supply"]),
        "game_active": bool(market["game_active"]), "claimant_count": len(market["claimants"]),
        "total_claimant_deficit": float(np.sum(market["deficits"])),
    }


def _pred(model: EpisodeForecastModel, method: str, client: str, series: pd.Series,
          origin: pd.Timestamp, h: int, coverage: float, cache=None) -> tuple[float, float, float]:
    key = (model.target, method, client, origin)
    if cache is not None and key in cache:
        mean, sd = cache[key]
    else:
        mean, sd = model.predict(method, client, series, origin)
    if h < 1 or h > model.H:
        return np.nan, np.nan, np.nan
    m = float(mean[h - 1]); s = float(sd[h - 1])
    if not np.isfinite(m) or not np.isfinite(s):
        return np.nan, np.nan, np.nan
    z = float(norm.ppf(0.5 + 0.5 * coverage))
    return m, m - z * s, m + z * s


def _integrated_distress(deficit: np.ndarray, essential: np.ndarray) -> np.ndarray:
    d = np.maximum(np.asarray(deficit, dtype=float), 0.0)
    l = np.maximum(np.asarray(essential, dtype=float), 0.0)
    out = np.zeros(len(d), dtype=float)
    # Forward-looking fraction of essential energy expected to be locally short.
    for h in range(len(d)):
        den = float(np.sum(l[h:]))
        out[h] = float(np.sum(d[h:]) / den) if den > 0.0 else 0.0
    return np.clip(out, 0.0, 1.0)


def _oracle_future(context: StepContext, origin: pd.Timestamp, H: int) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], list[np.ndarray], np.ndarray]:
    focal_d = []; focal_l = []; peer_d = []; peer_s = []; supplies = []
    peers = [c for c in context.measured_claimants if c != context.focal]
    for h in range(H):
        ts = origin + h * STEP
        fr = _row(context.frames[context.focal], ts)
        if fr is None:
            break
        fd = max(float(fr["critical_deficit_kwh"]), 0.0)
        fl = max(float(fr["critical_load_kwh"]), 0.0)
        pdv, psv = [], []
        for c in peers:
            rr = _row(context.frames[c], ts)
            if rr is None:
                continue
            dd = max(float(rr["critical_deficit_kwh"]), 0.0)
            ll = max(float(rr["critical_load_kwh"]), 0.0)
            if dd > 0.0:
                pdv.append(dd); psv.append(float(truthful_viability_signal(np.array([dd]), np.array([ll]))[0]))
        sup = 0.0
        for c in context.selected_clients:
            if c == context.focal: continue
            rr = _row(context.frames[c], ts)
            if rr is not None:
                sup += max(-float(rr["net_demand_kwh"]), 0.0)
        focal_d.append(fd); focal_l.append(fl); peer_d.append(np.asarray(pdv)); peer_s.append(np.asarray(psv)); supplies.append(sup)
    return np.asarray(focal_d), np.asarray(focal_l), peer_d, peer_s, np.asarray(supplies)


def strategic_step(context: StepContext, origin: pd.Timestamp, method: str,
                   control_horizon_steps: int, use_uncertainty: bool = True,
                   oracle: bool = False) -> dict:
    actual = _actual_market(context, origin)
    pos = _focal_position(actual, context.focal)
    if pos is None:
        return _base_result(actual, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    frow = _row(context.frames[context.focal], origin)
    fd0 = max(float(frow["critical_deficit_kwh"]), 0.0)
    fl0 = max(float(frow["critical_load_kwh"]), 0.0)

    H = int(control_horizon_steps)
    if oracle:
        fd, fl, peer_d, peer_s, supply = _oracle_future(context, origin, H)
    else:
        fd = [fd0]; fl = [fl0]; peer_d = []; peer_s = []; supply = [float(actual["supply"])]
        # Current state is observed exactly.
        pmask = np.arange(len(actual["deficits"])) != pos
        peer_d.append(actual["deficits"][pmask].copy())
        peer_s.append(actual["truthful_signals"][pmask].copy())
        zcov = context.forecasts.coverage
        models = context.forecasts.models
        series = context.forecasts.series
        peers = [c for c in context.measured_claimants if c != context.focal]
        for h in range(1, H):
            hh = h
            try:
                md = models["critical_deficit"]; ml = models["critical_load"]; mn = models["net_demand"]
                mfd, lfd, ufd = _pred(md, method, context.focal, series["critical_deficit"][context.focal], origin, hh, zcov, context.forecasts.cache)
                mfl, lfl, ufl = _pred(ml, method, context.focal, series["critical_load"][context.focal], origin, hh, zcov, context.forecasts.cache)
            except Exception:
                break
            if not np.all(np.isfinite([mfd, mfl])):
                break
            fd.append(max(mfd, 0.0)); fl.append(max(mfl, 0.0))
            pdv, psv = [], []
            for c in peers:
                if c not in md.clients or c not in ml.clients:
                    continue
                mdp, ldp, udp = _pred(md, method, c, series["critical_deficit"][c], origin, hh, zcov, context.forecasts.cache)
                mlp, llp, ulp = _pred(ml, method, c, series["critical_load"][c], origin, hh, zcov, context.forecasts.cache)
                if not np.all(np.isfinite([mdp, mlp])):
                    continue
                dd = max(udp if use_uncertainty else mdp, 0.0)
                load_for_signal = max(llp if use_uncertainty else mlp, np.finfo(float).eps)
                if dd > 0.0:
                    pdv.append(dd)
                    psv.append(float(np.clip(dd / load_for_signal, 0.0, 1.0)))
            peer_d.append(np.asarray(pdv, dtype=float)); peer_s.append(np.asarray(psv, dtype=float))
            sup = 0.0
            for c in context.selected_clients:
                if c == context.focal or c not in mn.clients:
                    continue
                mnm, lnm, unm = _pred(mn, method, c, series["net_demand"][c], origin, hh, zcov, context.forecasts.cache)
                if not np.isfinite(mnm):
                    continue
                upper_net = unm if use_uncertainty else mnm
                sup += max(-upper_net, 0.0)
            supply.append(sup)
        fd = np.asarray(fd, dtype=float); fl = np.asarray(fl, dtype=float); supply = np.asarray(supply, dtype=float)

    if len(fd) == 0:
        return minimum_service_step(context, origin)
    private_path = _integrated_distress(fd, fl)
    floors = np.zeros(len(fd), dtype=float)
    for h in range(len(fd)):
        floors[h], _ = minimum_service_preserving_signal(fd[h], peer_d[h], peer_s[h], supply[h])
    plan = plan_minimum_distortion_opacity(floors, private_path)
    signal = float(plan.signal[0])
    d, served, unserved = _allocate_actual(actual, context.focal, signal)
    result = _base_result(actual, d, signal, served, unserved,
                          float(truthful_viability_signal(np.array([fd0]), np.array([fl0]))[0]),
                          float(private_path[0]))
    result.update({
        "planned_covariance": plan.covariance,
        "planned_target_covariance": plan.target_covariance,
        "planned_distortion_l2": plan.distortion_l2,
        "planned_steps": len(plan.signal),
        "planned_game_active_steps": plan.active_steps,
        "service_floor_current": float(floors[0]),
    })
    return result
