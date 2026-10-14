import numpy as np
from fedscope.game.controller import _integrated_distress


def test_integrated_distress_is_energy_fraction_not_max():
    d=np.array([1.,0.,1.]); l=np.array([2.,2.,2.])
    z=_integrated_distress(d,l)
    assert np.isclose(z[0],2/6)
    assert np.isclose(z[1],1/4)
    assert np.isclose(z[2],1/2)
