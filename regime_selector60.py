"""Regime-adaptive 60% trade win-rate research challenger.

This is not a future prediction guarantee. All statistics are updated ONLY from
historical trades whose exit date is strictly before the scored signal date.
The training/validation policy is fixed ex ante. Diagnostic data is never used
to select thresholds or a policy.
"""
from collections import defaultdict
from math import sqrt, isfinite
from profit_selector51 import stats, TRAIN, VALID, END

MIN_COMPLETED = 45
MIN_LOCAL = 18
TARGET_PCT = 60.0

def regime(row):
    """Signal-time regime features only; no realized outcome."""
    try:
        slope=float(row["signal_ma20_ma60_pct"])
        atr=float(row["signal_atr_pct"])
        rsi=float(row["signal_rsi"])
        adx=float(row["signal_adx"])
        if not all(isfinite(v) for v in (slope,atr,rsi,adx)):return None
        return (int(slope>0),int(atr>=3.0),int(adx>=25.0),int(rsi>=50.0))
    except (KeyError,TypeError,ValueError):return None

def _lower(w,n):
    if n==0:return 0.0
    z=1.28;p=w/n
    return (p+z*z/(2*n)-z*sqrt(p*(1-p)/n + z*z/(4*n*n)))/(1+z*z/n)

def score_all(trades, market=None, min_history=MIN_COMPLETED):
    """Online prequential with exit-matured updates; forecasts never use current outcome."""
    items=[]
    for i,row in enumerate(trades):
        if market is not None and row.get("market")!=market:continue
        signal=str(row.get("signal_date",""))[:10]
        exit_date=str(row.get("exit_date",""))[:10]
        try:net=float(row["net_pct"])
        except (KeyError,TypeError,ValueError):continue
        if len(signal)!=10 or len(exit_date)!=10 or exit_date<signal or not isfinite(net):continue
        items.append((i,{**row,"signal_date":signal,"exit_date":exit_date,"net_pct":net}))
    chron=sorted(items,key=lambda x:(x[1]["signal_date"],x[0]))
    settled=sorted(items,key=lambda x:(x[1]["exit_date"],x[0]))
    global_n=global_w=0
    groups=defaultdict(lambda:[0,0,0.])
    pending=[]
    index=0
    for _,row in chron:
        now=row["signal_date"]
        while index<len(settled) and settled[index][1]["exit_date"]<now:
            old=settled[index][1];index+=1
            global_n+=1;global_w+=int(old["net_pct"]>0)
            k=regime(old)
            if k is not None:
                s=groups[k];s[0]+=1;s[1]+=int(old["net_pct"]>0);s[2]+=old["net_pct"]
        key=regime(row)
        if key is None or global_n<min_history:continue
        n,w,net=groups[key]
        prior=(global_w+2)/(global_n+4)
        # Effective count of each regime shrinks to a conservative pooled prior.
        expected=(w+24*prior)/(n+24)
        # A local break-even net edge is mandatory before considering the trade.
        edge=(net+24*((global_w/global_n)-0.5))/(n+24)
        quality=_lower(w,n) if n>=MIN_LOCAL else 0.0
        pending.append({**row,"regime_key":key,"probability":expected,
                        "local_n":n,"lower_win_bound":quality,"local_net_avg":net/n if n else None,
                        "prior":prior,"observed":int(row["net_pct"]>0)})
    return pending

# Predeclared competing entry policies. No diagnostic-window optimization.
POLICIES={
 "base":lambda r: True,
 "p55":lambda r: r["probability"]>=0.55 and r["local_n"]>=MIN_LOCAL,
 "p60":lambda r: r["probability"]>=0.60 and r["local_n"]>=MIN_LOCAL,
 "p60_local_profit":lambda r:r["probability"]>=0.60 and r["local_n"]>=MIN_LOCAL
    and r["local_net_avg"] is not None and r["local_net_avg"]>0,
 "p65_local_profit":lambda r:r["probability"]>=0.65 and r["local_n"]>=MIN_LOCAL
    and r["local_net_avg"] is not None and r["local_net_avg"]>0,
}
def report(trades,market=None):
    scored=score_all(trades,market)
    windows={
      "train":[r for r in scored if r["signal_date"]<TRAIN and r["exit_date"]<TRAIN],
      "validation":[r for r in scored if TRAIN<=r["signal_date"] and r["exit_date"]<VALID],
      "diagnostic":[r for r in scored if VALID<=r["signal_date"] and r["exit_date"]<END]
    }
    candidate={}
    for name,filter_fn in POLICIES.items():
        candidate[name]={}
        for window,rows in windows.items():
            sub=[r for r in rows if filter_fn(r)]
            m=stats(sub)
            m["coverage_pct"]=round(100*len(sub)/len(rows),3) if rows else None
            candidate[name][window]=m
    # Select only from training with sufficient observed trades, then freeze it.
    qualified=[]
    for name,windows_scores in candidate.items():
        m=windows_scores["train"]
        if name=="base" or m["n"]<35 or m["mean_net_pct"]<=0:continue
        # Stability and sufficient sample size before maximising lower bound.
        sub=[r for r in windows["train"] if POLICIES[name](r)]
        lcb=_lower(sum(r["observed"] for r in sub),len(sub))
        qualified.append((lcb,m["mean_net_pct"],name))
    if qualified:
        qualified.sort(key=lambda x:(-x[0],-x[1],x[2]))
        winner=qualified[0][2]
    else:winner=None
    validation=candidate[winner]["validation"] if winner else None
    accepted=(winner is not None and validation["n"]>=30
              and validation["win_pct"]>=TARGET_PCT
              and validation["mean_net_pct"]>0
              and (validation["profit_factor"] is None and validation["win_pct"]==100
                   or validation["profit_factor"] is not None and validation["profit_factor"]>1))
    return {"market":market,"target_win_pct":TARGET_PCT,"scored":len(scored),
            "status":"VALIDATION_PASS_RESEARCH_ONLY" if accepted else "REJECTED",
            "selected_policy":winner,"validation_pass":accepted,"live_approved":False,
            "by_policy":candidate,
            "note":"Inspected historical diagnostics are NOT independent forward tests."}
