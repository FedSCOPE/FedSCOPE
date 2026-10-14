import pandas as pd

from fedscope.publication_audit import independent_dependence


def test_publication_audit_reads_clean_v21_outputs(tmp_path):
    out = tmp_path / "results_final"
    out.mkdir()
    ts = pd.date_range("2026-01-01", periods=8, freq="15min", tz="UTC")
    target = [0.1, 0.2, 0.35, 0.3, 0.55, 0.65, 0.7, 0.9]
    rows = []
    signals = {
        "truthful": target,
        "minimum_service": [0.0, 0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.7],
        "strategic_executed_opacity": [0.2, 0.1, 0.35, 0.2, 0.4, 0.6, 0.5, 0.7],
    }
    for policy, sig in signals.items():
        for t, s, d in zip(ts, sig, target):
            rows.append({
                "timestamp": t,
                "policy": policy,
                "signal": s,
                "game_active": True,
                "common_anticipated_private_horizon": d,
            })
    pd.DataFrame(rows).to_csv(out / "policy_step_results.csv", index=False)

    result = independent_dependence(tmp_path)
    assert set(result.policy) == set(signals)
    assert set(result.configuration) == {"results_final"}
    assert (result.n == 8).all()
    assert "normalized_hsic_reduction_vs_min_pct" in result.columns
