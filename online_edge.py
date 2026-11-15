"""Disk-backed, bounded-state, no-lookahead market-specific trade calibration.

Research only. SQLite sorts arbitrarily large input iterators without retaining
all trades in Python memory. Processing is chronological. A trade's outcome
updates the prediction model ONLY after its exit date is strictly before the
next signal date. This module cannot guarantee any win rate.
"""
from collections import defaultdict
import heapq
import json
import math
import os
import sqlite3
import tempfile

MIN_HISTORY=35
MIN_LOCAL=12
SMOOTH=15.0
# Three frozen research policies; NEVER optimize over already inspected holdout.
POLICIES={
    "online_coverage":{"prob":0.49,"edge":0.10},
    "online_balanced":{"prob":0.54,"edge":0.25},
    "online_high_confidence":{"prob":0.60,"edge":0.45},
}

def regime_key(t):
    """All inputs can be known at signal time (no outcome fields)."""
    try:
        rsi=float(t["signal_rsi"])
        atr=float(t["signal_atr_pct"])
        slope=float(t["signal_ma20_ma60_pct"])
        if not all(math.isfinite(v) for v in (rsi,atr,slope)):
            return None
        return (int(rsi>=55.0),int(atr>=3.0),int(slope>=0.0))
    except (TypeError,ValueError,KeyError):
        return None

class RunningEdge:
    """O(1) update and prediction; constant memory per discrete regime."""
    def __init__(self):
        self.total=[0.0,0.0,0.0]  # count, winning count, net sum
        self.local=defaultdict(lambda:[0.0,0.0,0.0])
    def add(self,t):
        key=regime_key(t)
        try:
            net=float(t["net_pct"])
            if key is None or not math.isfinite(net):return False
        except (KeyError,TypeError,ValueError):return False
        for state in (self.total,self.local[key]):
            state[0]+=1;state[1]+=float(net>0);state[2]+=net
        return True
    def estimate(self,t):
        key=regime_key(t)
        if key is None or self.total[0]<MIN_HISTORY:return None
        global_count,global_win,global_net=self.total
        n,w,net=self.local[key]
        if n<MIN_LOCAL:return None
        # Hierarchical Beta shrinkage: local probability regularized toward
        # market-wide empirical prior. No maximum rate is presumed.
        global_p=(global_win+2)/(global_count+4)
        p=(w+SMOOTH*global_p)/(n+SMOOTH)
        base_return=global_net/global_count
        edge=(net+SMOOTH*base_return)/(n+SMOOTH)
        return {"predicted_win_pct":round(100*p,3),
                "predicted_net_pct":round(edge,4),
                "prior_local_trades":int(n),
                "prior_market_trades":int(global_count),
                "market_base_win_pct":round(100*global_p,3)}

def _date(text):
    return str(text)[:10]

def score_stream(trades, market, *, db_path=None):
    """Score an iterable of historical hypothetical trades with bounded RAM.

    Output selected trades are lists for ordinary backtest compatibility;
    to score huge datasets, supply a callback or drain results by batches.
    Input is spooled to temporary SQLite, indexed by signal and exit date.
    """
    owned=db_path is None
    if owned:
        fd,db_path=tempfile.mkstemp(prefix="stock_radar_stream_",suffix=".sqlite3")
        os.close(fd)
    conn=sqlite3.connect(db_path)
    try:
        conn.execute("CREATE TABLE IF NOT EXISTS stream_trades (id INTEGER PRIMARY KEY, signal TEXT, exit TEXT, ticker TEXT, data TEXT)")
        conn.execute("DELETE FROM stream_trades")
        staged=0
        batch=[]
        for item in trades:
            if item.get("market")!=market:continue
            sig=_date(item.get("signal_date",""))
            end=_date(item.get("exit_date",""))
            if len(sig)!=10 or len(end)!=10 or end<sig:continue
            batch.append((sig,end,str(item.get("ticker","")),json.dumps(item,ensure_ascii=False,default=str)))
            if len(batch)>=1000:
                conn.executemany("INSERT INTO stream_trades(signal,exit,ticker,data) VALUES (?,?,?,?)",batch)
                staged+=len(batch);batch.clear()
        if batch:
            conn.executemany("INSERT INTO stream_trades(signal,exit,ticker,data) VALUES (?,?,?,?)",batch)
            staged+=len(batch)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_stream_signal ON stream_trades(signal,ticker,id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_stream_exit ON stream_trades(exit,id)")
        conn.commit()
        # Two independently indexed cursors avoid sorting all trade records in RAM.
        signals=conn.execute("SELECT signal,data FROM stream_trades ORDER BY signal,ticker,id")
        exits=conn.execute("SELECT exit,data FROM stream_trades ORDER BY exit,id")
        finished=iter(exits)
        next_exit=next(finished,None)
        model=RunningEdge()
        selected={k:[] for k in POLICIES}
        diagnostics={"population":staged,"insufficient":0,"scored":0,
                     "by_policy":{k:0 for k in POLICIES},
                     "calibration_bins":{str(k):{"count":0,"wins":0,"sum_net":0.0} for k in range(10)}}
        for signal,data in signals:
            # Strict before, never same-day settlement or later outcome.
            while next_exit is not None and next_exit[0]<signal:
                model.add(json.loads(next_exit[1]))
                next_exit=next(finished,None)
            row=json.loads(data)
            estimate=model.estimate(row)
            if estimate is None:
                diagnostics["insufficient"]+=1;continue
            diagnostics["scored"]+=1
            bin_id=min(9,int(estimate["predicted_win_pct"]//10))
            b=diagnostics["calibration_bins"][str(bin_id)]
            b["count"]+=1
            b["wins"]+=int(float(row["net_pct"])>0)
            b["sum_net"]+=float(row["net_pct"])
            for name,limits in POLICIES.items():
                if estimate["predicted_win_pct"]>=100*limits["prob"] and estimate["predicted_net_pct"]>=limits["edge"]:
                    selected[name].append({**row,**{"online_"+k:v for k,v in estimate.items()}})
                    diagnostics["by_policy"][name]+=1
        for bucket in diagnostics["calibration_bins"].values():
            count=bucket["count"]
            bucket["actual_win_rate_pct"]=round(100*bucket["wins"]/count,2) if count else None
            bucket["actual_avg_net_pct"]=round(bucket["sum_net"]/count,3) if count else None
        return selected,diagnostics
    finally:
        conn.close()
        if owned:
            try:os.remove(db_path)
            except OSError:pass
