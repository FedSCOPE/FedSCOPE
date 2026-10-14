import numpy as np
from fedscope.evaluation.leakage import distance_correlation


def test_distance_correlation_detects_nonlinear_dependence():
    x=np.linspace(-1,1,101); y=x*x
    assert distance_correlation(x,y)>0.3
