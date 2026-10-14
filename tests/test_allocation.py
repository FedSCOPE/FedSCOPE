import numpy as np
from fedscope.game.allocation import game_active, weighted_proportional_fair_allocation


def test_game_active_exact_condition():
    assert game_active(np.array([2.0,3.0]),4.0)
    assert not game_active(np.array([2.0,3.0]),5.0)
    assert not game_active(np.array([2.0,0.0]),1.0)
    assert not game_active(np.array([2.0,3.0]),0.0)


def test_zero_signal_is_opt_out_in_active_game():
    a=weighted_proportional_fair_allocation(np.array([5.0,5.0]),np.array([0.0,1.0]),5.0)
    assert np.isclose(a[0],0.0)
    assert np.isclose(a[1],5.0)


def test_equal_weights_equal_deficits_split_supply():
    a=weighted_proportional_fair_allocation(np.array([10.0,10.0]),np.array([1.0,1.0]),6.0)
    assert np.allclose(a,[3.0,3.0],rtol=0,atol=1e-10)
