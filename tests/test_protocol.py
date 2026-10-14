import numpy as np
import pandas as pd
from fedscope.forecast.eb_random_effects import ClientSuffStats, EpisodeForecastModel


def test_protocol_never_claims_raw_rows_sent():
    cutoff=pd.Timestamp('2020-01-01',tz='UTC')
    x=np.array([[1.,0.],[1.,1.],[1.,2.],[1.,3.],[1.,4.],[1.,5.]])
    def st(c,k):
        Y=np.column_stack([k*x[:,1],(k+.1)*x[:,1]])
        return ClientSuffStats(c,cutoff,len(x),x.T@x,x.T@Y,(Y*Y).sum(0),len(x))
    m=EpisodeForecastModel.fit('x',cutoff,(1,),2,{'a':st('a',1.),'b':st('b',1.2)})
    assert all(r['server_receives_raw_rows'] is False for r in m.protocol_rows)
