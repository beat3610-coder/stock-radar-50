"""Research-only broad candidate generation independent of legacy buy filters.

Every tenth observation evaluates a hypothetical executable next-open purchase,
including weak/negative trend states. These are labelled using historical
stop-first exits to train a classifier; generated outcomes are NEVER consulted
when generating the signal, risk plan, or features.
"""
import math
import pandas as pd
import numpy as np
from trade_execution import entry_quality
from audit_300 import simulate_exit

BROAD_STEP=10
HOLD=10
WARMUP=240

def broad_candidates_from_computed(x,ticker,market,end_date="2026-11-01",cost_pct=0.35):
    rows=[]
    last_exit=-1
    for i in range(WARMUP,len(x)-HOLD,BROAD_STEP):
        date=str(pd.Timestamp(x.index[i]).date())
        if date>=end_date:break
        if i<=last_exit:continue
        r=x.iloc[i]
        close=float(r.Close); nxt=float(x.Open.iloc[i+1])
        names=("ATR","MA20","MA60","MA200","RSI","ADX","VOL20","Volume")
        try:
            if not all(np.isfinite(float(r[k])) for k in names):continue
            atr=float(r.ATR);v20=float(r.VOL20);ma60=float(r.MA60)
            past_close=float(x.Close.iloc[i-5])
            if min(close,nxt,atr,v20,ma60,past_close)<=0:continue
            if not 0.004<=atr/close<=0.085:continue
            if float(r.Volume)<=0:continue
            # Unlike technical BUY rules, permit falling MA200, RSI weakness,
            # flat MACD, and downtrends so the learner observes losses too.
            stop=close-max(0.022*close,1.65*atr)
            stop=max(stop,close*0.93)
            target=close+1.8*(close-stop)
            check=entry_quality(nxt,stop,target,reference=close,cost_pct=cost_pct)
            if not check["accepted"]:continue
            outcome=simulate_exit(x,i,nxt,stop,target,max_days=HOLD,cost_pct=cost_pct)
            if not outcome:continue
            rows.append({"market":market,"ticker":ticker,"strategy":"broad_market_ml",
                "signal_date":date,
                "entry_date":str(pd.Timestamp(x.index[i+1]).date()),
                "exit_date":str(pd.Timestamp(x.index[outcome["exit_index"]]).date()),
                "entry":nxt,"stop":stop,"take":target,
                "entry_net_rr":check["net_rr"],
                "entry_net_upside_pct":check["net_upside_pct"],
                "signal_atr_pct":100*atr/close,
                "signal_ma20_ma60_pct":100*(float(r.MA20)/ma60-1),
                "signal_rsi":float(r.RSI),
                "signal_adx":float(r.ADX),
                "signal_volume_ratio":float(r.Volume)/v20,
                "signal_5d_return_pct":100*(close/past_close-1),
                "entry_gap_pct":100*(nxt/close-1),
                "score":float(r.RSI),**outcome})
            last_exit=outcome["exit_index"]
        except (KeyError,TypeError,ValueError,OverflowError,ZeroDivisionError):
            continue
    return rows
