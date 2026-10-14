from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

METHOD = "strategic_executed_opacity"
BASELINE = "minimum_service"


def _dcor(run_dir: Path, policy: str) -> float:
    df = pd.read_csv(run_dir / "leakage_global.csv")
    row = df[(df.policy == policy) & df.target.isin(["common_anticipated_horizon", "anticipated"])]
    if len(row) != 1:
        raise RuntimeError(f"Missing unique anticipated-distress leakage row for {policy} in {run_dir}")
    return float(row.iloc[0].distance_correlation)


def _reduction(run_dir: Path) -> float:
    base = _dcor(run_dir, BASELINE)
    method = _dcor(run_dir, METHOD)
    return 100.0 * (base - method) / base


def main() -> None:
    ap = argparse.ArgumentParser(description="Verify key FedSCOPE reproduction metrics.")
    ap.add_argument("--project-root", type=Path, default=Path("."))
    args = ap.parse_args()

    root = args.project_root.resolve()
    ref = json.loads((root / "reference_results" / "expected_metrics.json").read_text())
    failures: list[str] = []

    def check(name: str, actual: float, expected: float, atol: float) -> None:
        ok = bool(np.isclose(float(actual), float(expected), rtol=0.0, atol=atol))
        print(f"{'PASS' if ok else 'FAIL'}  {name}: {actual:.12g}  reference={expected:.12g}")
        if not ok:
            failures.append(name)

    primary = root / "results_final"
    suite = root / "paper_suite_final"
    audits = root / "publication_audits"
    manifest = json.loads((primary / "run_manifest.json").read_text())
    exp = ref["primary"]

    episodes = int(manifest["episodes"]["evaluated"])
    active = int(manifest["episodes"]["game_active_steps"])
    for name, actual, expected in [
        ("episodes", episodes, exp["episodes"]),
        ("game-active states", active, exp["game_active_steps"]),
    ]:
        ok = actual == expected
        print(f"{'PASS' if ok else 'FAIL'}  {name}: {actual}  reference={expected}")
        if not ok:
            failures.append(name)

    summary = pd.read_csv(primary / "primary_summary.csv").set_index("policy")
    check("minimum-service served kWh", summary.loc[BASELINE, "game_served_kwh"], exp["minimum_service_served_kwh"], 5e-6)
    check("FedSCOPE served kWh", summary.loc[METHOD, "game_served_kwh"], exp["fedscope_served_kwh"], 5e-6)
    check("minimum-service anticipated dCor", _dcor(primary, BASELINE), exp["minimum_service_anticipated_dcor"], 5e-6)
    check("FedSCOPE anticipated dCor", _dcor(primary, METHOD), exp["fedscope_anticipated_dcor"], 5e-6)

    h = ref["horizon_anticipated_dcor_reduction_pct"]
    check("6 h dCor reduction (%)", _reduction(suite / "horizon_6h"), h["horizon_6h"], 0.15)
    check("12 h dCor reduction (%)", _reduction(suite / "horizon_12h"), h["horizon_12h"], 0.15)
    check("24 h dCor reduction (%)", _reduction(primary), h["primary_24h"], 0.15)

    independent = pd.read_csv(audits / "independent_privacy_metrics.csv")
    independent = independent[independent.configuration == "results_final"].set_index("policy")
    a = ref["independent_audits"]
    for col, base_key, method_key in [
        ("normalized_hsic", "minimum_service_hsic", "fedscope_hsic"),
        ("mi_knn_k3", "minimum_service_mi_k3", "fedscope_mi_k3"),
        ("mi_knn_k5", "minimum_service_mi_k5", "fedscope_mi_k5"),
        ("mi_knn_k10", "minimum_service_mi_k10", "fedscope_mi_k10"),
        ("mi_knn_k20", "minimum_service_mi_k20", "fedscope_mi_k20"),
    ]:
        check(f"minimum-service {col}", independent.loc[BASELINE, col], a[base_key], 5e-6)
        check(f"FedSCOPE {col}", independent.loc[METHOD, col], a[method_key], 5e-6)

    comm = pd.read_csv(audits / "federated_communication_evidence.csv")
    row = comm[comm.target == "ALL"].iloc[0]
    ratio_ok = float(row.raw_equivalent_to_payload_ratio) > 2000.0
    raw_ok = not bool(row.server_receives_any_raw_rows)
    print(f"{'PASS' if ratio_ok else 'FAIL'}  raw-equivalent/payload ratio: {float(row.raw_equivalent_to_payload_ratio):.1f} (>2000 expected)")
    print(f"{'PASS' if raw_ok else 'FAIL'}  server raw-row exposure: {bool(row.server_receives_any_raw_rows)}")
    if not ratio_ok: failures.append("payload ratio")
    if not raw_ok: failures.append("raw-row exposure")

    if failures:
        print("\nReproduction check FAILED:")
        for item in failures:
            print(f"  - {item}")
        raise SystemExit(1)
    print("\nReproduction check PASSED.")


if __name__ == "__main__":
    main()
