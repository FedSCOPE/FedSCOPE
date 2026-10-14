import numpy as np
from fedscope.game.opacity import minimum_service_preserving_signal, focal_served, plan_minimum_distortion_opacity


def test_minimum_signal_preserves_max_service():
    pd=np.array([4.0,6.0]); ps=np.array([0.5,0.8]); supply=5.0
    s,mx=minimum_service_preserving_signal(7.0,pd,ps,supply)
    got=focal_served(7.0,pd,s,ps,supply)
    assert got >= mx - 1e-10
    if s>1e-8:
        lower=focal_served(7.0,pd,max(s-1e-6,0),ps,supply)
        assert lower <= mx + 1e-8


def test_opacity_never_below_service_floor():
    floor=np.array([0.2,0.4,0.1,0.5]); private=np.array([0.1,0.8,0.3,0.9])
    p=plan_minimum_distortion_opacity(floor,private)
    assert np.all(p.signal>=floor-1e-12)
    assert np.all(p.signal<=1+1e-12)


def test_constant_private_returns_floor():
    floor=np.array([0.2,0.4,0.1]); private=np.ones(3)
    p=plan_minimum_distortion_opacity(floor,private)
    assert np.allclose(p.signal,floor)
