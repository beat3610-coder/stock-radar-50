"""Offline regression checks for chronological target calibration and stop-first exits."""
import numpy as np
import pandas as pd
from trade_plan import scenario_from_history

n=420
idx=pd.bdate_range("2024-01-01",periods=n)
rng=np.random.default_rng(42)
close=100*np.exp(np.cumsum(rng.normal(0.0003,0.012,size=n)))
op=np.r_[close[0],close[:-1]]*(1+rng.normal(0,0.002,size=n))
hi=np.maximum(op,close)*(1+0.006)
lo=np.minimum(op,close)*(1-0.006)
x=pd.DataFrame({"Open":op,"High":hi,"Low":lo,"Close":close},index=idx)
x["MA20"]=x.Close.rolling(20).mean()
x["MA60"]=x.Close.rolling(60).mean()
plan=scenario_from_history(x,float(close[-1])*0.94)
assert plan["history_samples"]>=15,plan
assert "학습" in plan["basis"] and "평가" in plan["basis"],plan
assert plan["target"] is None or plan["target"]>close[-1],plan
assert plan["target"] is None or plan["planned_rr"]>=0,plan
assert -100<plan["historical_net_expectancy_pct"]<100,plan

# No fabricated target when a valid risk reference is missing.
no_stop=scenario_from_history(x,None)
assert no_stop["target"] is None
print("PASS: holdout target evaluation and missing-stop safeguards")
