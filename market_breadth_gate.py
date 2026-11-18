"""Market-breadth regime admission gate, research-only.

Uses only close prices available on the signal date. Built once from the 100
loaded constituents for each market. This is NOT the actual capitalization-
weighted KOSPI index; it is equal-stock breadth and has survivorship bias.
Selections are made on pre-November-2025 training only. Validation and the
already-inspected April-October-2026 diagnostic never tune guard choice.
"""
from collections import defaultdict
from math import isfinite
import numpy as np
import pandas as pd
from profit_selector51 import TRAIN, VALID, END, stats

# Predeclared hypotheses, no diagnostic-period optimization.
GUARDS={
 "breadth_50":lambda r: r["market_breadth60_pct"]>=50,
 "breadth_60":lambda r: r["market_breadth60_pct"]>=60,
 "breadth_60_positive_5d":lambda r: r["market_breadth60_pct"]>=60 and r["market_mean_5d_pct"]>0,
 "breadth_65_lowvol":lambda r: r["market_breadth60_pct"]>=65 and r["market_mean_atr_pct"]<=3.5,
 "breadth_55_momentum":lambda r: r["market_breadth60_pct"]>=55 and r["market_mean_5d_pct"]>=1.0,
}
def build_breadth(frames,min_stocks=30):
    aggregated=defaultdict(list)
    for ticker,frame in frames.items():
        try:
            close=pd.to_numeric(frame["Close"],errors="coerce").sort_index()
            if len(close)<80:continue
            ma=close.rolling(60,min_periods=60).mean()
            r5=100*(close/close.shift(5)-1)
            atr_proxy=100*(pd.to_numeric(frame["High"])-pd.to_numeric(frame["Low"]))/close
            # Same-day close, 60-day mean, past 5 closes, intraday range.
            for dt,cl,m,change,vol in zip(close.index,close.to_numpy(),
                                           ma.to_numpy(),r5.to_numpy(),atr_proxy.to_numpy()):
                if not all(isfinite(float(x)) for x in (cl,m,change,vol)) or m<=0:continue
                aggregated[str(pd.Timestamp(dt).date())].append((cl>m,float(change),float(vol)))
        except (KeyError,TypeError,ValueError,OverflowError):
            continue
    result={}
    for day,records in aggregated.items():
        if len(records)<min_stocks:continue
        result[day]={"market_breadth60_pct":100*sum(r[0] for r in records)/len(records),
            "market_mean_5d_pct":float(np.mean([r[1] for r in records])),
            "market_mean_atr_pct":float(np.mean([r[2] for r in records])),
            "breadth_stocks":len(records)}
    return result

def _selection(rows):
    n=len(rows)
    return stats(rows),n

def audit(trades,frames,market=None):
    breadth=build_breadth(frames)
    usable=[]
    for row in trades:
        if market is not None and row.get("market")!=market:continue
        sig=str(row.get("signal_date",""))[:10]
        exitdate=str(row.get("exit_date",""))[:10]
        if len(sig)!=10 or len(exitdate)!=10 or exitdate<sig or sig>=END:continue
        if sig not in breadth:continue
        try:
            net=float(row["net_pct"])
            if not isfinite(net):continue
        except (KeyError,TypeError,ValueError):continue
        usable.append({**row,**breadth[sig],"signal_date":sig,"exit_date":exitdate,"net_pct":net})
    windows={
      "training":[r for r in usable if r["signal_date"]<TRAIN and r["exit_date"]<TRAIN],
      "validation":[r for r in usable if TRAIN<=r["signal_date"] and r["exit_date"]<VALID],
      "diagnostic":[r for r in usable if VALID<=r["signal_date"] and r["exit_date"]<END]
    }
    baseline={k:stats(v) for k,v in windows.items()}
    policies={}
    for name,guard in GUARDS.items():
        policies[name]={}
        for period,rows in windows.items():
            picked=[r for r in rows if guard(r)]
            policies[name][period]={**stats(picked),
                "coverage_pct":round(100*len(picked)/len(rows),2) if rows else None}
    result={"market":market,"data_source":"equal_weight_constituent_breadth_NOT_index",
      "target_win_pct":60.0,"status":"NO_TRAINING_GUARD",
      "selected_guard":None,"baseline":baseline,
      "all_guards":policies,"usable":len(usable),
      "breadth_dates":len(breadth),"live_approved":False,
      "caveat":"2026 current-member survivorship, diagnostic previously inspected"}
    eligible=[]
    b=baseline["training"]
    for name,d in policies.items():
        s=d["training"]
        if s["n"]<45 or s["mean_net_pct"] is None or s["mean_net_pct"]<=0:continue
        if b["win_pct"] is None or s["win_pct"]<=b["win_pct"]:continue
        if b["mean_net_pct"] is None or s["mean_net_pct"]<=b["mean_net_pct"]:continue
        # Penalize tiny samples; no post-hoc tuning by validation/diagnostic.
        score=s["mean_net_pct"]-1.5/np.sqrt(s["n"])
        eligible.append((score,s["n"],name))
    if not eligible:return result
    eligible.sort(key=lambda v:(-v[0],-v[1],v[2]))
    _,_,name=eligible[0]
    v=policies[name]["validation"]
    accepted=(v["n"]>=30 and v["win_pct"] is not None and v["win_pct"]>=60
        and v["mean_net_pct"]>0
        and (v["profit_factor"] is None and v["win_pct"]==100 or
             v["profit_factor"] is not None and v["profit_factor"]>1))
    result.update({"selected_guard":name,
        "status":"VALIDATION_PASS_RESEARCH_ONLY" if accepted else "VALIDATION_REJECTED",
        "validation_pass":accepted,
        "selected_training":policies[name]["training"],
        "selected_validation":v,
        "selected_diagnostic":policies[name]["diagnostic"]})
    return result
