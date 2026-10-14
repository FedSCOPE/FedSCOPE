from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

STRATEGIC = "strategic_executed_opacity"
MINIMUM = "minimum_service"
TRUTHFUL = "truthful"


def _style() -> None:
    plt.rcParams.update({
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def _save(fig, out: Path, name: str) -> None:
    fig.tight_layout()
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(out / f"{name}.png", dpi=450, bbox_inches="tight")
    plt.close(fig)


def _leakage_reduction(run_dir: Path, metric: str) -> float:
    leak = pd.read_csv(run_dir / "leakage_global.csv")
    q = leak[leak.target.isin(["common_anticipated_horizon", "anticipated"])].set_index("policy")
    a = float(q.loc[MINIMUM, metric])
    b = float(q.loc[STRATEGIC, metric])
    return 100.0 * (a - b) / a if a > 0 else np.nan


def run(project_root: Path, audit_dir: Path, out: Path) -> None:
    _style()
    out.mkdir(parents=True, exist_ok=True)
    primary = project_root / "results_final"
    suite = project_root / "paper_suite_final"

    steps = pd.read_csv(primary / "policy_step_results.csv")
    steps["timestamp"] = pd.to_datetime(steps["timestamp"], utc=True)
    fed = steps[(steps.policy == STRATEGIC) & steps.game_active.astype(bool)].copy()

    # 1) Seasonal frequency + scarcity severity.
    fed["month"] = fed.timestamp.dt.tz_localize(None).dt.to_period("M").astype(str)
    fed["scarcity_pct"] = 100.0 * fed["supply"] / fed["total_claimant_deficit"]
    monthly = fed.groupby("month").agg(
        game_steps=("timestamp", "size"),
        median_scarcity_pct=("scarcity_pct", "median"),
        median_claimants=("claimant_count", "median"),
    )
    full_months = pd.period_range(pd.Period(monthly.index.min()), pd.Period(monthly.index.max()), freq="M").astype(str)
    monthly = monthly.reindex(full_months)
    monthly["game_steps"] = monthly.game_steps.fillna(0)
    monthly = monthly.reset_index().rename(columns={"index": "month"})
    x = np.arange(len(monthly))
    fig, ax = plt.subplots(figsize=(7.2, 3.45))
    ax.bar(x, monthly.game_steps, alpha=0.82, label="Game-active states")
    ax.set_ylabel("Game-active 15-min states")
    ax.set_xlabel("Month")
    ax.set_xticks(x, [m[2:] for m in monthly.month], rotation=45, ha="right")
    ax.grid(axis="y", alpha=0.22)
    ax2 = ax.twinx()
    ax2.plot(x, monthly.median_scarcity_pct, marker="o", linewidth=1.55, label="Median supply / claimant deficit")
    ax2.set_yscale("log")
    ax2.set_ylabel("Median transferable supply / claimant deficit (%)")
    ax2.scatter(x, monthly.median_scarcity_pct, s=35 + 22 * (monthly.median_claimants - 2), alpha=0.55)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper right")
    ax.set_title("Seasonal frequency and severity of genuine scarcity competition")
    _save(fig, out, "fig_scarcity_timeline")

    # 2) Federated performance profile; IQR across clients, marker size = median tau^2.
    fm = pd.read_csv(primary / "forecast_metrics.csv")
    eb = pd.read_csv(primary / "eb_sharing.csv")
    targets = [("critical_deficit", "Essential deficit"), ("critical_load", "Essential load"), ("net_demand", "Net demand")]
    fig, ax = plt.subplots(figsize=(7.2, 3.55))
    hsteps = {0.25: 1, 1.0: 4, 6.0: 24, 12.0: 48, 24.0: 96}
    for target, label in targets:
        q = fm[fm.target == target]
        p = q.pivot_table(index=["client", "horizon_hours"], columns="method", values="mae").reset_index()
        p["rel"] = 100.0 * (p["federated"] - p["local"]) / p["local"]
        g = p.groupby("horizon_hours")["rel"]
        hx = np.array(sorted(p.horizon_hours.unique()), dtype=float)
        med = np.array([g.get_group(h).median() for h in hx])
        q25 = np.array([g.get_group(h).quantile(.25) for h in hx])
        q75 = np.array([g.get_group(h).quantile(.75) for h in hx])
        tau = eb[eb.target == target].groupby("horizon_steps").tau2_reml.median()
        sizes = np.array([45 + 300 * np.sqrt(float(tau.loc[hsteps[h]])) for h in hx])
        ax.plot(hx, med, marker="o", linewidth=1.5, label=label)
        ax.fill_between(hx, q25, q75, alpha=0.12)
        ax.scatter(hx, med, s=sizes, alpha=0.55)
    ax.axhline(0, linewidth=0.8)
    ax.set_xscale("log")
    ax.set_xticks([0.25, 1, 6, 12, 24], ["0.25", "1", "6", "12", "24"])
    ax.set_xlabel("Forecast horizon (h)")
    ax.set_ylabel("Federated MAE change relative to local OLS (%)")
    ax.set_title("Federated empirical-Bayes forecasting: client dispersion and adaptive sharing")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="best")
    _save(fig, out, "fig_federated_profile")

    # 3) Operating map of the executed controller.
    truth = steps[(steps.policy == TRUTHFUL) & steps.game_active.astype(bool)][["timestamp", "signal"]].set_index("timestamp")["signal"]
    fed = fed.set_index("timestamp").sort_index()
    fed["truth_signal"] = truth.reindex(fed.index)
    fed["upper"] = np.maximum(fed.service_floor_current, fed.truth_signal)
    fed["headroom"] = fed.upper - fed.service_floor_current
    fed["lift"] = fed.signal - fed.service_floor_current
    fed["upper_executed"] = fed["lift"] > 1e-9
    fig, ax = plt.subplots(figsize=(7.2, 3.75))
    lo = fed[~fed.upper_executed]; hi = fed[fed.upper_executed]
    sz_lo = 10 + 12 * (lo.claimant_count - 2); sz_hi = 18 + 15 * (hi.claimant_count - 2)
    ax.scatter(lo.headroom, lo.common_anticipated_private_horizon, s=sz_lo, alpha=0.18, marker="o", label="Executed floor")
    ax.scatter(hi.headroom, hi.common_anticipated_private_horizon, s=sz_hi, alpha=0.75, marker="x", label="Executed truth-bounded upper action")
    ax.set_xlabel(r"Instantaneous service-preserving headroom $u_t-\ell_t$")
    ax.set_ylabel(r"Causal anticipated distress $D_t^{(H)}$")
    ax.set_title("Executed controller operating map under genuine scarcity")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right")
    _save(fig, out, "fig_operating_map")

    # 4) Unified cross-metric/cross-horizon robustness.
    indep = pd.read_csv(audit_dir / "independent_privacy_metrics.csv")
    configs = [(suite / "horizon_6h", "paper_suite_final__horizon_6h", "6 h"),
               (suite / "horizon_12h", "paper_suite_final__horizon_12h", "12 h"),
               (primary, "results_final", "24 h")]
    metric_labels = ["|Pearson|", "|Spearman|", "dCor", "HSIC", "MI k=3", "MI k=5", "MI k=10", "MI k=20"]
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    x = np.arange(len(metric_labels))
    for run_dir, cfg, label in configs:
        vals = [_leakage_reduction(run_dir, "pearson_abs"), _leakage_reduction(run_dir, "spearman_abs"), _leakage_reduction(run_dir, "distance_correlation")]
        q = indep[indep.configuration == cfg].set_index("policy")
        for col in ["normalized_hsic", "mi_knn_k3", "mi_knn_k5", "mi_knn_k10", "mi_knn_k20"]:
            a = float(q.loc[MINIMUM, col]); b = float(q.loc[STRATEGIC, col])
            vals.append(100.0 * (a - b) / a if a > 0 else np.nan)
        ax.plot(x, vals, marker="o", linewidth=1.65, label=label)
    ax.axhline(0, linewidth=0.8)
    ax.set_xticks(x, metric_labels, rotation=18, ha="right")
    ax.set_ylim(-5, 105)
    ax.set_ylabel("Reduction relative to minimum-service signaling (%)")
    ax.set_xlabel("Dependence / information-exposure metric")
    ax.set_title("Opacity improvement across horizons and independent audit metrics")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper right", ncol=3)
    _save(fig, out, "fig_privacy_robustness")

    # 5) Focal heterogeneity explained by executable headroom.
    head = pd.read_csv(audit_dir / "opacity_headroom_by_focal.csv")
    name_map = {
        "results_final": "Industrial 3",
        "paper_suite_final__focal_DE_KN_residential1": "Residential 1",
        "paper_suite_final__focal_DE_KN_residential3": "Residential 3",
        "paper_suite_final__focal_DE_KN_residential4": "Residential 4",
        "paper_suite_final__focal_DE_KN_residential6": "Residential 6",
    }
    head["label"] = head.configuration.map(name_map).fillna(head.configuration)
    size = 65 + 180 * np.sqrt(head.game_active_steps / head.game_active_steps.max())
    fig, ax = plt.subplots(figsize=(7.2, 3.65))
    ax.scatter(head.mean_executed_lift_above_floor, head.dcor_reduction_vs_min_pct, s=size, alpha=0.8)
    for _, r in head.iterrows():
        ax.annotate(r.label, (r.mean_executed_lift_above_floor, r.dcor_reduction_vs_min_pct), xytext=(5, 5), textcoords="offset points", fontsize=8)
    ax.axhline(0, linewidth=0.8)
    ax.set_xlabel(r"Mean executed lift above exact service floor $\mathbb{E}|s_t-\ell_t|$")
    ax.set_ylabel("Anticipated dCor reduction vs. minimum service (%)")
    ax.set_title("Client-dependent opacity gain follows executable service-preserving headroom")
    ax.grid(alpha=0.25)
    _save(fig, out, "fig_headroom_mechanism")

    print(f"[FedSCOPE figures] wrote {out.resolve()}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", type=Path, default=Path("."))
    ap.add_argument("--audit-dir", type=Path, default=Path("publication_audits"))
    ap.add_argument("--output", type=Path, default=Path("publication_figures"))
    args = ap.parse_args()
    run(args.project_root, args.audit_dir, args.output)


if __name__ == "__main__":
    main()
