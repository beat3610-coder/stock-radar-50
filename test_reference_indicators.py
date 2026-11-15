import numpy as np
import pandas as pd

from reference_indicators import add_reference_indicators,wilder_average


ix=pd.bdate_range("2025-01-01",periods=120)
base=np.linspace(100,140,len(ix))+np.sin(np.arange(len(ix))/4)
frame=pd.DataFrame({
    "Open":base-.2,"High":base+1.2,"Low":base-1.0,
    "Close":base,"Volume":100_000+np.arange(len(ix))*500,
},index=ix)

out=add_reference_indicators(frame)
required=("RSI_WILDER","ATR_WILDER","ADX_WILDER","CMF20",
          "DONCHIAN20_POS","ROC20","BBWIDTH20_PCT")
assert all(name in out for name in required)
assert out.RSI_WILDER.dropna().between(0,100).all()
assert (out.ATR_WILDER.dropna()>0).all()
assert (out.BBWIDTH20_PCT.dropna()>=0).all()

# A final-bar change must not alter any earlier indicator value.
changed=frame.copy()
changed.iloc[-1,changed.columns.get_loc("Close")]+=10
changed.iloc[-1,changed.columns.get_loc("High")]+=10
before=out.iloc[-2][list(required)]
after=add_reference_indicators(changed).iloc[-2][list(required)]
assert np.allclose(before,after,equal_nan=True)

sample=pd.Series([np.nan,1.,2.,3.,4.])
rma=wilder_average(sample,3)
assert np.isclose(rma.iloc[3],2.0)
assert np.isclose(rma.iloc[4],(2.0*2+4.0)/3)
print("reference indicator tests passed")
