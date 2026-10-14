import numpy as np
import pandas as pd
from fedscope.data.opsd import _interval_energy


def test_cumulative_meter_differencing_requires_consecutive_timestamp():
    idx=pd.to_datetime(['2020-01-01T00:00:00Z','2020-01-01T00:15:00Z','2020-01-01T01:00:00Z'])
    raw=pd.DataFrame({'m':[10.,12.,15.]},index=idx)
    consecutive=raw.index.to_series().diff().eq(pd.Timedelta(minutes=15))
    out=_interval_energy(raw,'m',consecutive)
    assert np.isnan(out.iloc[0]); assert out.iloc[1]==2.; assert np.isnan(out.iloc[2])
