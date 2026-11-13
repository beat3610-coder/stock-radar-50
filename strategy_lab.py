"""Chronological research lab for entry strategies.

HISTORICAL DATA USED HERE HAS BEEN SEEN THROUGH 2026-11; the 2026-04+
holdout is a reproducible regression diagnostic, not a pristine independent
holdout. Future unseen paper records are mandatory before live certification.

No optimization over holdout. Market-specific models are chosen ONLY from
development trades, checked on a fixed validation window, then reported on
the diagnostic holdout WITHOUT tuning to the holdout.
"""
from datetime import datetime, timezone
from pathlib import Path
import json
import math
import numpy as np
import pandas as pd
from engine import compute
from providers import get_prices
from universe import get_universe
from audit_300 import simulate_exit, metrics
from portfolio_audit import replay

TRAIN_END="2025-11-01"
VALID_END="2026-04-01"
HOLDOUT_END="2026-11-01"
COST_PCT=0.35
MIN_TRAIN_TRADES=20
MIN_VALID_TRADES=5
LOOKBACK=220
STEP=2
HOLD=10
PARAMS={
 "trend_breakout":{"rr":2.0,"stop_atr":1.8},
 "trend_pullback":{"rr":1.7,"stop_atr":1.7},
 "momentum_reentry":{"rr":1.8,"stop_atr":1.8},
}
def _finite(*values):
    return all(v is not None and np.isfinite(v) for v in values)

def qualifying(row,prev,market,variant,prior_high):
    p=float(row.Close)
    atr=float(row.ATR)
    if not _finite(row.MA20,row.MA60,row.MA200,row.RSI,row.MACDH,row.ADX,row.PDI,row.MDI,atr,row.VOL20):
        return False
    if atr<=0 or row.VOL20<=0:return False
    volatility=atr/p
    if volatility<0.006 or volatility>0.075:return False
    # Determine buying conditions strictly from signal-day and prior bars.
    if variant=="trend_breakout":
        return (p>row.MA20>row.MA60>row.MA200 and p>prior_high
            and 50<=row.RSI<=76 and row.MACDH>0
            and row.Volume/row.VOL20>=1.2)
    if variant=="trend_pullback":
        return (p>row.MA60>row.MA200 and p>row.MA20
            and 43<=row.RSI<=68 and
            float(prev.Close)<=float(prev.MA20)*1.018
            and p>float(prev.Close) and row.MACDH>float(prev.MACDH)
            and row.PDI>row.MDI)
    if variant=="momentum_reentry":
        return (p>row.MA60>row.MA200 and
            45<=row.RSI<=72 and row.MACDH>0
            and float(prev.MACDH)<=0 and p>row.MA20 and row.PDI>row.MDI)
    return False

def candidates_from_frame(frame,ticker,market):
    x=compute(frame)
    result={key:[] for key in PARAMS}
    for i in range(LOOKBACK,len(x)-HOLD,STEP):
        date=str(pd.Timestamp(x.index[i]).date())
        if date>=HOLDOUT_END:break
        r=x.iloc[i];prev=x.iloc[i-1]
        prior_high=float(x.High.iloc[i-20:i].max())
        close=float(r.Close)
        op=float(x.Open.iloc[i+1])
        atr=float(r.ATR)
        if not(_finite(close,op,atr) and op>0 and close>0):continue
        if abs(op/close-1)>0.025:continue
        for name,param in PARAMS.items():
            if not qualifying(r,prev,market,name,prior_high):continue
            # Stop and target are fixed at signal close. No future prices.
            stop=max(close*(1-0.095),close-param["stop_atr"]*atr)
            stop=min(stop,close*0.98)
            risk=op-stop
            if not(0<risk/op<=0.10):continue
            take=close+param["rr"]*(close-stop)
            if not(take>op and (take-op)/op>=0.02):continue
            outcome=simulate_exit(x,i,op,stop,take,max_days=HOLD,cost_pct=COST_PCT)
            if not outcome:continue
            trade={"market":market,"ticker":ticker,"strategy":name,
                "signal_date":date,"entry_date":str(pd.Timestamp(x.index[i+1]).date()),
                "exit_date":str(pd.Timestamp(x.index[outcome["exit_index"]]).date()),
                "entry":op,"stop":stop,"take":take,
                "score":float(r.RSI),**outcome}
            result[name].append(trade)
    # prevent same ticker stacking into its own previous trade
    for name in result:
        chosen=[];last_exit=""
        for t in sorted(result[name],key=lambda z:z["entry_date"]):
            if t["entry_date"]<=last_exit:continue
            chosen.append(t);last_exit=t["exit_date"]
        result[name]=chosen
    return x,result

def _interval(trades,part):
    if part=="train":return [t for t in trades if t["entry_date"]<TRAIN_END and t["exit_date"]<TRAIN_END]
    if part=="validate":return [t for t in trades if TRAIN_END<=t["entry_date"]<VALID_END and t["exit_date"]<VALID_END]
    return [t for t in trades if VALID_END<=t["entry_date"]<HOLDOUT_END and t["exit_date"]<HOLDOUT_END]

def train_choice(choices):
    """Development only. Don't reward single lucky huge wins."""
    scores={}
    for name,trades in choices.items():
        m=metrics(_interval(trades,"train"))
        n=m["trades"]
        pf=m["profit_factor"]
        avg=m["mean_net_pct"]
        # Baseline is 'no trade', not guaranteed positive expectancy.
        if n<MIN_TRAIN_TRADES or pf is None or avg is None or pf<1.15 or avg<=0:
            scores[name]=None
        else:
            # Penalize fragile small samples; cap one-off large returns.
            scores[name]=min(avg,3.0)*math.sqrt(n/(n+50))*min(pf,3)/3
    chosen=max((k for k,v in scores.items() if v is not None),
               key=lambda k:scores[k],default=None)
    return chosen,scores

def run(per_market=20):
    u=get_universe()
    res={"created_utc":datetime.now(timezone.utc).isoformat(),
        "type":"research_lab_v1",
        "method":"train-only selection, fixed validation screening, untouched-in-this-run diagnostic holdout",
        "train_end":TRAIN_END,"validation_end":VALID_END,"holdout_end":HOLDOUT_END,
        "selection_requires":{"train_trades":MIN_TRAIN_TRADES,"validation_trades":MIN_VALID_TRADES,
                              "train_pf":1.15,"validation_pf":1.15},
        "caveats":["Data from 2026 has been examined during earlier development; not truly independent",
                   "Current constituents create survivorship bias",
                   "Historical tickers' listing/delisting state not reconstructed",
                   "OHLC stop-first, next-day open, costs assumed 0.35%",
                   "Market-level capital has no live currency conversion",
                   "Strategy selection done on training only; validation used only as rejection gate",
                   "Never approve live trading without future independently observed paper results"],
        "markets":{}}
    for market in ("KR","KQ","US"):
        items=[x for x in u[market] if x[0] not in
               ({"KR":{"472150","0262B0"},"KQ":set(),"US":{"SOXL"}}[market])][:per_market]
        frames={}
        variants={k:[] for k in PARAMS}
        failures=[]
        for idx,(ticker,_,_,_) in enumerate(items,1):
            try:
                raw,_=get_prices(ticker,market)
                x,trades=candidates_from_frame(raw,ticker,market)
                frames[ticker]=raw
                for k in variants:variants[k].extend(trades[k])
            except Exception as exc:
                failures.append({"ticker":ticker,"error":str(exc)[:160]})
            if idx%5==0:
                print(f"LAB_PROGRESS {market} {idx}/{len(items)} loaded={len(frames)}",flush=True)
        leader,scores=train_choice(variants)
        train_metrics={k:metrics(_interval(v,"train")) for k,v in variants.items()}
        validation_metrics={k:metrics(_interval(v,"validate")) for k,v in variants.items()}
        test_metrics={k:metrics(_interval(v,"test")) for k,v in variants.items()}
        accepted=False
        if leader:
            vm=validation_metrics[leader]
            accepted=(vm["trades"]>=MIN_VALID_TRADES and
                      vm["mean_net_pct"] is not None and vm["mean_net_pct"]>0 and
                      vm["profit_factor"] is not None and vm["profit_factor"]>=1.15)
        selected=leader if accepted else None
        selected_trades=_interval(variants[selected],"test") if selected else []
        base=replay(selected_trades,frames,start=VALID_END,
                    initial_cash=10_000_000 if market in ("KR","KQ") else 100_000)
        stress=replay(selected_trades,frames,start=VALID_END,
                    initial_cash=10_000_000 if market in ("KR","KQ") else 100_000,
                    cost=0.007)
        def compact(r):return {k:v for k,v in r.items() if k not in ("daily_equity","executions")}
        res["markets"][market]={
            "requested":per_market,"loaded":len(frames),"errors":failures,
            "training_selected":leader,"training_scores":scores,
            "validation_accepted":accepted,
            "selected_for_diagnostic":selected,
            "train":train_metrics,"validation":validation_metrics,
            "holdout_all_variants_diagnostic_only":test_metrics,
            "holdout_selected_trades":len(selected_trades),
            "holdout_selected_metrics":metrics(selected_trades),
            "portfolio":compact(base),"stress_portfolio":compact(stress),
            "status":"NOT_LIVE_APPROVED"
        }
        print(f"LAB_MARKET {market} "+json.dumps({
            "selected":selected,"train":train_metrics.get(leader),
            "validation":validation_metrics.get(leader),
            "holdout":metrics(selected_trades),"account_return":base.get("total_return_pct"),
            "stress_return":stress.get("total_return_pct"),
            "loaded":len(frames)},default=str),flush=True)
    res["release_status"]="NOT_LIVE_APPROVED"
    return res

if __name__=="__main__":
    report=run()
    Path("data").mkdir(exist_ok=True)
    Path("data/strategy-lab-latest.json").write_text(
         json.dumps(report,ensure_ascii=False,indent=2,default=str),encoding="utf-8")
    print("LAB_RESULT "+json.dumps(report,ensure_ascii=False,default=str),flush=True)
