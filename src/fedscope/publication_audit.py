from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from sklearn.feature_selection import mutual_info_regression
except Exception:  # pragma: no cover
    mutual_info_regression = None

STRATEGIC_POLICY = "strategic_executed_opacity"
BASELINE_POLICY = "minimum_service"
TRUTHFUL_POLICY = "truthful"


def _rbf_gram_1d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float).reshape(-1, 1)
    d2 = (x - x.T) ** 2
    if len(x) < 2:
        return np.ones((len(x), len(x)), dtype=float)
    tri = np.sqrt(d2[np.triu_indices(len(x), 1)])
    positive = tri[tri > 0]
    sigma = float(np.median(positive)) if len(positive) else 1.0
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = 1.0
    return np.exp(-d2 / (2.0 * sigma * sigma))


def normalized_hsic(x: np.ndarray, y: np.ndarray) -> float:
    """Deterministic normalized HSIC audit using median-heuristic RBF kernels.

    This statistic is *not* used by the FedSCOPE controller.  It is reported as
    an independent nonlinear audit of the executed dCor-optimized policy.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 3 or np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return 0.0
    k = _rbf_gram_1d(x)
    l = _rbf_gram_1d(y)
    kc = k - k.mean(axis=0, keepdims=True) - k.mean(axis=1, keepdims=True) + k.mean()
    lc = l - l.mean(axis=0, keepdims=True) - l.mean(axis=1, keepdims=True) + l.mean()
    den = math.sqrt(float(np.sum(kc * kc) * np.sum(lc * lc)))
    return float(np.sum(kc * lc) / den) if den > 0 else 0.0


def mi_sensitivity(x: np.ndarray, y: np.ndarray, neighbors=(3, 5, 10, 20)) -> dict[str, float]:
    """kNN mutual-information sensitivity audit over prespecified k values."""
    if mutual_info_regression is None:
        return {f"mi_knn_k{k}": np.nan for k in neighbors}
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    xx, yy = x[m], y[m]
    out: dict[str, float] = {}
    for k in neighbors:
        if len(xx) <= k + 2 or np.allclose(xx, xx[0]) or np.allclose(yy, yy[0]):
            out[f"mi_knn_k{k}"] = 0.0
            continue
        out[f"mi_knn_k{k}"] = float(
            mutual_info_regression(
                xx.reshape(-1, 1), yy, n_neighbors=int(k), random_state=0
            )[0]
        )
    return out


def _normal_config_dirs(root: Path) -> list[Path]:
    dirs: list[Path] = []
    primary = root / "results_final"
    if (primary / "policy_step_results.csv").exists():
        dirs.append(primary)
    suite = root / "paper_suite_final"
    if suite.exists():
        for d in sorted(p for p in suite.iterdir() if p.is_dir()):
            if (d / "policy_step_results.csv").exists():
                dirs.append(d)
    return dirs


def _config_label(d: Path, root: Path) -> str:
    if d.name == "results_final":
        return "results_final"
    return f"paper_suite_final__{d.name}"

def _load_common_direct(d: Path) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(d / "policy_step_results.csv")
    if "common_anticipated_private_horizon" not in df:
        raise ValueError(f"Missing common anticipated target in {d / 'policy_step_results.csv'}")
    if "game_active" not in df:
        raise ValueError(f"Missing game_active in {d / 'policy_step_results.csv'}")
    df = df[df.game_active.astype(bool)].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    strategic = df[df.policy == STRATEGIC_POLICY].copy().set_index("timestamp").sort_index()
    if strategic.empty:
        raise ValueError(f"No {STRATEGIC_POLICY} rows found in {d}")
    target = strategic["common_anticipated_private_horizon"].astype(float)
    return df, target


def independent_dependence(root: Path) -> pd.DataFrame:
    records: list[dict] = []
    for d in _normal_config_dirs(root):
        df, target = _load_common_direct(d)
        cfg = _config_label(d, root)
        for policy in [TRUTHFUL_POLICY, BASELINE_POLICY, STRATEGIC_POLICY]:
            g = df[df.policy == policy].copy()
            g["timestamp"] = pd.to_datetime(g["timestamp"], utc=True)
            s = g.set_index("timestamp")["signal"].reindex(target.index).astype(float)
            m = np.isfinite(s.to_numpy()) & np.isfinite(target.to_numpy())
            rec = {
                "configuration": cfg,
                "policy": policy,
                "n": int(m.sum()),
                "normalized_hsic": normalized_hsic(s.to_numpy(), target.to_numpy()),
            }
            rec.update(mi_sensitivity(s.to_numpy(), target.to_numpy()))
            records.append(rec)
    out = pd.DataFrame(records)
    if out.empty:
        return out
    base = out[out.policy == BASELINE_POLICY].set_index("configuration")
    for metric in ["normalized_hsic", "mi_knn_k3", "mi_knn_k5", "mi_knn_k10", "mi_knn_k20"]:
        mapping = base[metric].to_dict()
        reductions = []
        for cfg, value in zip(out.configuration, out[metric]):
            b = mapping.get(cfg, np.nan)
            reductions.append(100.0 * (b - value) / b if np.isfinite(b) and b > 0 else np.nan)
        out[f"{metric}_reduction_vs_min_pct"] = reductions
    return out


def opacity_headroom(root: Path) -> pd.DataFrame:
    records: list[dict] = []
    for d in _normal_config_dirs(root):
        cfg = _config_label(d, root)
        if not (cfg == "results_final" or "focal_" in cfg):
            continue
        df, _ = _load_common_direct(d)
        st = df[df.policy == STRATEGIC_POLICY].copy()
        tr = df[df.policy == TRUTHFUL_POLICY].copy()
        st["timestamp"] = pd.to_datetime(st.timestamp, utc=True)
        tr["timestamp"] = pd.to_datetime(tr.timestamp, utc=True)
        st = st.set_index("timestamp").sort_index()
        tr = tr.set_index("timestamp").sort_index()
        truth = tr.signal.reindex(st.index).astype(float)
        floor_col = "service_floor_current" if "service_floor_current" in st.columns else "signal"
        floor = st[floor_col].astype(float)
        upper = np.maximum(floor.to_numpy(), truth.to_numpy())
        headroom = upper - floor.to_numpy()
        lift = st.signal.to_numpy(float) - floor.to_numpy()

        leak = pd.read_csv(d / "leakage_global.csv")
        target_names = ["common_anticipated_horizon", "anticipated"]
        l = leak[leak.target.isin(target_names)].set_index("policy")
        if BASELINE_POLICY in l.index and STRATEGIC_POLICY in l.index:
            dcor_min = float(l.loc[BASELINE_POLICY, "distance_correlation"])
            dcor_fedscope = float(l.loc[STRATEGIC_POLICY, "distance_correlation"])
            dcor_red = 100.0 * (dcor_min - dcor_fedscope) / dcor_min if dcor_min > 0 else np.nan
        else:
            dcor_red = dcor_min = dcor_fedscope = np.nan

        records.append({
            "configuration": cfg,
            "game_active_steps": int(len(st)),
            "mean_service_preserving_headroom": float(np.mean(headroom)),
            "median_service_preserving_headroom": float(np.median(headroom)),
            "fraction_positive_headroom": float(np.mean(headroom > 1e-10)),
            "mean_executed_lift_above_floor": float(np.mean(np.abs(lift))),
            "fraction_executed_above_floor": float(np.mean(np.abs(lift) > 1e-10)),
            "dcor_minimum_service": dcor_min,
            "dcor_fedscope": dcor_fedscope,
            "dcor_reduction_vs_min_pct": dcor_red,
        })
    return pd.DataFrame(records)


def federated_systems_evidence(heavy_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    primary = heavy_root / "results_final"
    proto_file = primary / "federated_protocol.csv"
    eb_file = primary / "eb_sharing.csv"
    fm_file = primary / "forecast_metrics.csv"
    if not (proto_file.exists() and eb_file.exists() and fm_file.exists()):
        raise FileNotFoundError(
            "Expected results_final/{federated_protocol,eb_sharing,forecast_metrics}.csv"
        )

    p = pd.read_csv(proto_file)
    comm = []
    groups = [("ALL", p)] + [(str(target), g) for target, g in p.groupby("target")]
    for target, g in groups:
        payload = float(g.payload_scalars.sum())
        raw = float(g.raw_scalar_equivalent.sum())
        comm.append({
            "target": target,
            "records": int(len(g)),
            "payload_scalars": payload,
            "raw_scalar_equivalent": raw,
            "raw_equivalent_to_payload_ratio": raw / payload if payload > 0 else np.nan,
            "server_receives_any_raw_rows": bool(g.server_receives_raw_rows.astype(bool).any()),
        })
    comm_df = pd.DataFrame(comm)

    e = pd.read_csv(eb_file)
    eb = e.groupby("target").tau2_reml.agg([
        ("n", "size"), ("median_tau2", "median"),
        ("mean_tau2", "mean"), ("min_tau2", "min"), ("max_tau2", "max")
    ]).reset_index()
    q = e.groupby("target").tau2_reml.quantile([0.25, 0.75]).unstack()
    q.columns = ["tau2_q25", "tau2_q75"]
    eb = eb.merge(q.reset_index(), on="target", how="left")
    eb["fraction_tau2_near_zero"] = [
        float(np.mean(e.loc[e.target == t, "tau2_reml"].to_numpy() < 1e-10))
        for t in eb.target
    ]

    f = pd.read_csv(fm_file)
    z = f[f.method.isin(["local", "pooled", "federated"])].pivot_table(
        index=["target", "client", "horizon_steps"], columns="method", values="mae"
    ).dropna()
    wins = []
    for target, g in z.groupby(level=0):
        best = g.min(axis=1)
        wins.append({
            "target": target,
            "client_horizon_groups": int(len(g)),
            "federated_best_groups": int(np.sum(np.isclose(g.federated, best, atol=1e-12, rtol=0))),
            "federated_not_worse_than_local": int(np.sum(g.federated <= g.local)),
            "federated_not_worse_than_pooled": int(np.sum(g.federated <= g.pooled)),
            "mean_relative_mae_change_vs_local_pct": float(100.0 * np.mean((g.federated - g.local) / g.local)),
            "median_relative_mae_change_vs_local_pct": float(100.0 * np.median((g.federated - g.local) / g.local)),
        })
    return comm_df, eb, pd.DataFrame(wins)


def _write_report(output: Path, indep: pd.DataFrame, head: pd.DataFrame, comm: pd.DataFrame) -> None:
    lines = ["# FedSCOPE Publication Audit", ""]
    primary = indep[(indep.configuration == "results_final") & indep.policy.isin([BASELINE_POLICY, STRATEGIC_POLICY])]
    if len(primary) == 2:
        mn = primary[primary.policy == BASELINE_POLICY].iloc[0]
        nw = primary[primary.policy == STRATEGIC_POLICY].iloc[0]
        lines += [
            "## Independent privacy metrics on the primary run", "",
            f"- Normalized HSIC: {mn.normalized_hsic:.6f} -> {nw.normalized_hsic:.6f} ({100*(mn.normalized_hsic-nw.normalized_hsic)/mn.normalized_hsic:.1f}% reduction).",
            f"- kNN MI, k=3: {mn.mi_knn_k3:.6f} -> {nw.mi_knn_k3:.6f} ({100*(mn.mi_knn_k3-nw.mi_knn_k3)/mn.mi_knn_k3:.1f}% reduction).",
            f"- kNN MI, k=5: {mn.mi_knn_k5:.6f} -> {nw.mi_knn_k5:.6f} ({100*(mn.mi_knn_k5-nw.mi_knn_k5)/mn.mi_knn_k5:.1f}% reduction).",
            f"- kNN MI, k=10: {mn.mi_knn_k10:.6f} -> {nw.mi_knn_k10:.6f} ({100*(mn.mi_knn_k10-nw.mi_knn_k10)/mn.mi_knn_k10:.1f}% reduction).",
            f"- kNN MI, k=20: {mn.mi_knn_k20:.6f} -> {nw.mi_knn_k20:.6f} ({100*(mn.mi_knn_k20-nw.mi_knn_k20)/mn.mi_knn_k20:.1f}% reduction).",
            "",
            "HSIC and MI are post-hoc audits; they are never used by the executed-history controller.", "",
        ]
    if len(head) >= 3:
        corr = head[["dcor_reduction_vs_min_pct", "mean_executed_lift_above_floor"]].corr().iloc[0, 1]
        lines += [
            "## Client-dependent opacity headroom", "",
            f"The descriptive correlation between focal dCor gain and mean executed lift above the exact service floor is {corr:.3f} across the available focal rotations.",
            "This is a mechanism diagnostic, not a population-level significance claim.", "",
        ]
    allc = comm[comm.target == "ALL"]
    if len(allc):
        r = allc.iloc[0]
        lines += [
            "## Federated systems evidence", "",
            f"The sufficient-statistic scalar payload is {r.raw_equivalent_to_payload_ratio:.1f}x smaller than the scalar count of the corresponding centralized design/target rows.",
            f"Any server-side raw-row exposure recorded by the protocol audit: {bool(r.server_receives_any_raw_rows)}.", "",
        ]
    lines += [
        "## Scope", "",
        "These audits strengthen dependence and federated-system evidence without changing the core controller or retraining forecasts.",
        "They do not establish differential privacy, information-theoretic independence, or distribution-network power-flow feasibility.",
    ]
    (output / "PUBLICATION_AUDIT_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def run(heavy_root: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    indep = independent_dependence(heavy_root)
    indep.to_csv(output / "independent_privacy_metrics.csv", index=False)
    head = opacity_headroom(heavy_root)
    head.to_csv(output / "opacity_headroom_by_focal.csv", index=False)
    comm, eb, wins = federated_systems_evidence(heavy_root)
    comm.to_csv(output / "federated_communication_evidence.csv", index=False)
    eb.to_csv(output / "empirical_bayes_sharing_evidence.csv", index=False)
    wins.to_csv(output / "federated_forecast_win_counts.csv", index=False)
    _write_report(output, indep, head, comm)

    manifest = {
        "heavy_forecast_retraining": False,
        "source_mode": "direct_outputs",
        "uses_stored_executed_actions": True,
        "independent_metric": "normalized HSIC with median-heuristic RBF kernels",
        "mi_sensitivity_neighbors": [3, 5, 10, 20],
        "federated_evidence": [
            "payload/raw-equivalent scalar ratio",
            "REML tau2 summaries",
            "client-horizon forecast win counts",
        ],
    }
    (output / "PUBLICATION_AUDIT_MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Post-hoc publication audits for completed FedSCOPE runs.")
    ap.add_argument("--heavy-root", type=Path, default=Path("."), help="Project root containing results_final/ and paper_suite_final/.")
    ap.add_argument("--output", type=Path, default=Path("publication_audits"))
    args = ap.parse_args()
    run(args.heavy_root, args.output)
    print(f"[FedSCOPE publication audit] complete: {args.output.resolve()}")


if __name__ == "__main__":
    main()
