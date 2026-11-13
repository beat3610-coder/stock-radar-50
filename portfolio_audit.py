"""Daily mark-to-market, cash-constrained, simultaneous-position audit.

This is a portfolio REPLAY over independently generated trade candidates.
All orders use next-session open; exit prices follow OHLC stop-first logic.
No overnight cash borrowing, no fractional shares and no portfolio leverage.
"""
from collections import defaultdict
from math import floor
import numpy as np
import pandas as pd

INITIAL_CASH=100_000.0
COST=0.0035
MAX_POSITIONS=5
MAX_WEIGHT=0.20
RISK_BUDGET=0.01

def replay(trades, frames, *, start="2026-04-01", initial_cash=INITIAL_CASH,
           cost=COST, max_positions=MAX_POSITIONS, max_weight=MAX_WEIGHT,
           risk_budget=RISK_BUDGET):
    """Replay only trades with their signal and entry after fixed start date.

    Daily portfolio curve uses official-observed OHLC input frame closes,
    not compounded per-trade returns. Returns are in a unitless single
    market currency; do not mix KRW/USD in one capital ledger.
    """
    if not frames or initial_cash<=0:
        return {"valid":False,"reason":"no prices"}
    close={}
    for ticker,frame in frames.items():
        if frame is None or frame.empty:continue
        close[ticker]=pd.Series(frame["Close"].astype(float).values,
                                index=pd.to_datetime(frame.index).normalize())
    if not close:return {"valid":False,"reason":"no valid OHLC"}
    dates=sorted(set().union(*(set(s.index) for s in close.values())))
    dates=[d for d in dates if d>=pd.Timestamp(start)]
    if len(dates)<20:return {"valid":False,"reason":"under 20 portfolio dates"}
    entries=defaultdict(list)
    for t in trades:
        if t.get("entry_date", "")>=start and t.get("exit_date") and t.get("ticker") in close:
            entries[pd.Timestamp(t["entry_date"])].append(t)
    cash=float(initial_cash)
    holdings={}
    last={}
    curve=[]
    executions=[]
    for date in dates:
        # Latest available price, including across exchange holidays.
        for ticker,s in close.items():
            if date in s.index:
                price=float(s.loc[date])
                if np.isfinite(price) and price>0:last[ticker]=price
        # Existing exits at their actual stop/take/time fill, including gaps.
        for ticker,item in list(holdings.items()):
            if date>=item["exit_date"]:
                cash+=item["shares"]*item["exit_price"]*(1-cost/2)
                gross=item["shares"]*(item["exit_price"]-item["entry"])
                fees=item["shares"]*(item["entry"]+item["exit_price"])*cost/2
                executions.append({"ticker":ticker,"entry_date":str(item["entry_date"].date()),
                                   "exit_date":str(date.date()),"shares":item["shares"],
                                   "net_cash":round(gross-fees,2),
                                   "net_pct":round(100*(item["exit_price"]*(1-cost/2)/
                                                       (item["entry"]*(1+cost/2))-1),3)})
                del holdings[ticker]
        # Only admit trades whose next-open entry is today. Use the signal
        # score for fixed deterministic priority, never future returns.
        todo=sorted(entries.get(date,[]),key=lambda t:(-float(t.get("score") or 0),str(t["ticker"])))
        for t in todo:
            ticker=t["ticker"]
            if ticker in holdings or len(holdings)>=max_positions:continue
            op=float(t["entry"]);stop=float(t["stop"])
            if not(0<stop<op and np.isfinite(op) and np.isfinite(stop)):continue
            equity=cash+sum(x["shares"]*last.get(k,x["entry"]) for k,x in holdings.items())
            budget=min(cash,max_weight*equity)
            unit_risk=(op-stop)+op*cost
            shares=min(floor(budget/(op*(1+cost/2))),
                       floor(equity*risk_budget/unit_risk))
            if shares<1:continue
            outflow=shares*op*(1+cost/2)
            if outflow>cash:continue
            cash-=outflow
            holdings[ticker]={**t,"shares":shares,
                              "entry_date":date,
                              "exit_date":pd.Timestamp(t["exit_date"])}
            # Entry-day stops or take profits can occur after the opening fill.
            if pd.Timestamp(t["exit_date"])==date:
                item=holdings.pop(ticker)
                ep=float(item["exit_price"])
                cash+=shares*ep*(1-cost/2)
                executions.append({"ticker":ticker,"entry_date":str(date.date()),
                                   "exit_date":str(date.date()),"shares":shares,
                                   "net_cash":round(shares*((ep-op)-cost*(ep+op)/2),2),
                                   "net_pct":round(100*(ep*(1-cost/2)/(op*(1+cost/2))-1),3)})
        value=cash+sum(item["shares"]*last.get(k,item["entry"]) for k,item in holdings.items())
        curve.append((str(date.date()),float(value)))
    vals=np.array([v for _,v in curve],dtype=float)
    peak=np.maximum.accumulate(np.r_[initial_cash,vals])[1:]
    dd=100*(1-vals/peak)
    return {"valid":True,"initial_equity":initial_cash,
            "end_equity":round(float(vals[-1]),2),
            "total_return_pct":round(100*(float(vals[-1])/initial_cash-1),3),
            "max_drawdown_pct":round(float(max(0,dd.max())),3),
            "trades_executed":len(executions),"days":len(curve),
            "mark_to_market":True,
            "no_leverage":True,
            "max_concurrent_positions":max_positions,
            "max_position_weight":max_weight,
            "risk_budget_per_trade":risk_budget,
            "daily_equity":curve,
            "executions":executions,
            "benchmark_verified":False,
            "caveat":"Single-market simplified daily OHLC; no intraday cross-position allocation or market impact. No verified benchmark."}
