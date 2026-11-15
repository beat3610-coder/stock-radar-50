"""Offline tests: weak regime and overextended entries must not qualify."""
import numpy as np
import pandas as pd
from engine import compute
from strategy_lab import qualifying

n=340
idx=pd.bdate_range("2024-01-01",periods=n)
def frame(prices):
    prices=np.asarray(prices,dtype=float)
    op=prices*0.999
    return compute(pd.DataFrame({"Open":op,"High":prices*1.01,
        "Low":op*0.99,"Close":prices,"Volume":np.full(n,100000.)},
        index=idx))

# Recent prices may rebound while their 200-session trend is still declining.
falling=frame(np.linspace(200,100,n))
r=falling.iloc[-1].copy()
r["RSI"]=55;r["MACDH"]=1;r["ADX"]=30;r["PDI"]=30;r["MDI"]=10
assert not bool(r.MA200_UP20)
assert not qualifying(r,falling.iloc[-2],"KR","momentum_reentry",110)

# Strong price momentum does not allow chasing a price >8% above its MA20.
rising=frame(np.linspace(100,200,n))
r=rising.iloc[-1].copy()
r["RSI"]=55;r["MACDH"]=1;r["ADX"]=30;r["PDI"]=30;r["MDI"]=10
assert bool(r.MA200_UP20)
r["Close"]=float(r.MA20)*1.10
assert not qualifying(r,rising.iloc[-2],"US","momentum_reentry",150)

print("PASS: falling long average and overextended-price entries blocked")
