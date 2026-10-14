import numpy as np
import pandas as pd
from fedscope.forecast.eb_random_effects import ClientSuffStats, fit_local, EpisodeForecastModel


def make_stats(client, slope, cutoff, H=3):
    # Deterministic full-rank design with intercept + one lag.
    x=np.array([[1.,0.],[1.,1.],[1.,2.],[1.,3.],[1.,4.],[1.,5.]])
    Y=np.column_stack([1+slope*x[:,1]+0.01*h*x[:,1] for h in range(H)])
    G=x.T@x; R=x.T@Y; y2=(Y*Y).sum(0)
    return ClientSuffStats(client,cutoff,len(x),G,R,y2,len(x))


def test_local_sufficient_stats_fit_exact_linear_relation():
    st=make_stats('a',2.0,pd.Timestamp('2020-01-01',tz='UTC'))
    f=fit_local(st)
    assert np.allclose(f.beta[1,0],2.0,atol=1e-10)


def test_empirical_bayes_model_is_deterministic():
    cutoff=pd.Timestamp('2020-01-01',tz='UTC')
    stats={'a':make_stats('a',1.0,cutoff),'b':make_stats('b',1.2,cutoff),'c':make_stats('c',0.8,cutoff)}
    m1=EpisodeForecastModel.fit('x',cutoff,(1,),3,stats)
    m2=EpisodeForecastModel.fit('x',cutoff,(1,),3,stats)
    for c in stats:
        assert np.allclose(m1.coefficients[c].eb_beta,m2.coefficients[c].eb_beta)
        assert np.allclose(m1.coefficients[c].eb_tau2,m2.coefficients[c].eb_tau2)
