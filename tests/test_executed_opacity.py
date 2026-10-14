import numpy as np

from fedscope.evaluation.leakage import distance_correlation
from fedscope.game.executed_opacity import CausalExecutedDistanceOpacity


def test_incremental_candidate_matches_batch_distance_correlation():
    state = CausalExecutedDistanceOpacity()
    xs = [0.2, 1.0, 0.4, 0.7]
    ys = [0.8, 0.3, 0.5, 0.9]
    history_x = []
    history_y = []
    for x, y in zip(xs, ys):
        inc = state._candidate_dcor(x, y)
        batch = distance_correlation(np.asarray(history_x + [x]), np.asarray(history_y + [y]))
        assert np.isclose(inc, batch, atol=1e-12, rtol=1e-12)
        state._append(x, y)
        history_x.append(x)
        history_y.append(y)


def test_choice_is_service_preserving_endpoint_and_updates_history():
    state = CausalExecutedDistanceOpacity()
    d1 = state.choose(0.35, 0.7, 0.8)
    assert d1.signal == 0.35  # exact first-step tie -> minimum service floor
    assert d1.history_size_after == 1
    d2 = state.choose(0.4, 0.9, 0.2)
    assert d2.signal in (0.4, 0.9)
    assert d2.signal >= 0.4
    assert d2.history_size_after == 2


def test_undefined_private_target_falls_back_without_polluting_history():
    state = CausalExecutedDistanceOpacity()
    d = state.choose(0.42, 0.8, np.nan)
    assert d.signal == 0.42
    assert state.n == 0
