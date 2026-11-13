"""Next-open, stop-first, cost-aware diagnostic of trade-plan outcomes.

Selected current tickers = survivorship bias. No historical investor flows.
Do not treat this as a production-quality investment backtest.
"""
import json
from datetime import datetime,timezone
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from engine import compute,evaluate
from providers import get_prices
from universe import get_universe

# The 300 candidates are selected *before* any trade outcomes are computed.
SAMPLE_ALLOCATION={"KR":100,"KQ":100,"US":100}
MAX_WORKERS=4

def pick_300():
    universe=get_universe()
    selected=[]
    for market,n in SAMPLE_ALLOCATION.items():
        # Exclude the optional leveraged ETFs; sample ordinary companies only.
        stocks=[(market,code,name) for code,name,_,sector in universe[market]
                if code not in ("472150","0262B0","SOXL")]
        if len(stocks)<n:
            raise RuntimeError(f"{market}: 필요 {n}종목, 조회된 종목 {len(stocks)}종목")
        # Use each market's entire current 100-member universe (fixed before scoring).
        positions=np.linspace(0,min(99,len(stocks)-1),num=n,dtype=int)
        selected.extend(stocks[int(i)] for i in positions)
    assert len(selected)==300 and len({(m,c) for m,c,_ in selected})==300
    return selected,universe


HORIZON=10
TRADING_COST_PCT=0.35
STRIDE=10

def execute_path(df, i, entry, stop_pct, take_pct):
    """Trade from the next open, use actual following OHLC bars. Stop first if both touch."""
    stop=entry*(1-stop_pct/100)
    take=entry*(1+take_pct/100)
    reason="time"
    price=None
    for k in range(i+1,min(i+1+HORIZON,len(df))):
        bar=df.iloc[k]
        # Daily open gaps get the actual open, not optimistic stop-fill.
        if bar.Open<=stop:
            price=float(bar.Open);reason="gap_stop";break
        if bar.Open>=take:
            price=float(bar.Open);reason="gap_take";break
        if bar.Low<=stop:
            price=stop;reason="stop";break
        if bar.High>=take:
            price=take;reason="take";break
    if price is None:price=float(df.Close.iloc[min(i+HORIZON,len(df)-1)])
    gross=(price/entry-1)*100
    return {"net_return_pct":gross-TRADING_COST_PCT,"exit_reason":reason,
            "gross_return_pct":gross}

def execute_variant(df, i, entry, stop, take, mode):
    """Fixed variants evaluated with identical entries, gaps and costs.

    HOLD_10: hold 10 sessions.
    STOP_10: fixed historical stop, exit otherwise at 10th close.
    TAKE_STOP: existing take-first or stop-first strategy.
    HALF_TRAIL: half exit at target; remaining half trailing stop based only
    on completed bar closes (next day's stop effective), or time exit.
    """
    if mode=="TAKE_STOP":
        return execute_path(df,i,entry,100*(entry-stop)/entry,
                            100*(take-entry)/entry)["net_return_pct"]
    if mode=="HOLD_10":
        return 100*(float(df.Close.iloc[i+HORIZON])/entry-1)-TRADING_COST_PCT
    if mode=="STOP_10":
        for j in range(i+1,i+HORIZON+1):
            b=df.iloc[j]
            if b.Open<=stop:return 100*(float(b.Open)/entry-1)-TRADING_COST_PCT
            if b.Low<=stop:return 100*(stop/entry-1)-TRADING_COST_PCT
        return 100*(float(df.Close.iloc[i+HORIZON])/entry-1)-TRADING_COST_PCT
    if mode=="HALF_TRAIL":
        active_stop=stop
        first_half=None
        remaining=None
        trail_fraction=(entry-stop)/entry
        for j in range(i+1,i+HORIZON+1):
            b=df.iloc[j]
            if b.Open<=active_stop:
                remaining=float(b.Open)
                if first_half is None:first_half=remaining
                break
            if b.Low<=active_stop:
                remaining=active_stop
                if first_half is None:first_half=remaining
                break
            if first_half is None and b.High>=take:
                first_half=take
            # No same-bar upward trailing stop: update from completed close only.
            if first_half is not None:
                active_stop=max(active_stop,float(b.Close)*(1-trail_fraction))
        if remaining is None:remaining=float(df.Close.iloc[i+HORIZON])
        if first_half is None:first_half=remaining
        return 100*((first_half+remaining)/(2*entry)-1)-TRADING_COST_PCT
    raise ValueError("Unknown mode")

def test_ticker(market, code):
    df,_=get_prices(code,market)
    x=compute(df)
    samples=[]
    snapshots=[]
    for i in range(260,len(x)-HORIZON,STRIDE):
        data=x.iloc[:i+1]
        r=evaluate(data,market=market,flow=None)
        signal=r["signal"]
        # Directional audit covers EVERY historical decision, not just actual entries.
        next_open=float(x.Open.iloc[i+1])
        close_10d=float(x.Close.iloc[i+HORIZON])
        return_10d=100*(close_10d/next_open-1)-TRADING_COST_PCT if next_open>0 else float("nan")
        if np.isfinite(return_10d):
            category=("BUY" if signal in ("기술매수 후보·수급 미확인","매수 검토","강한 매수 검토")
                      else "SELL" if signal=="축소·매도 검토"
                      else "AVOID" if signal in ("하락 추세·매수 보류","과열·추격 주의")
                      else "HOLD")
            snapshots.append({"group":category,"signal":signal,"forward_10d_net_pct":return_10d})
        if signal not in ("기술매수 후보·수급 미확인","매수 검토","강한 매수 검토"):
            continue
        if r["stop_risk_pct"] is None or r["upside_pct"] is None:
            continue
        # Entry is the first knowable next session open. If a large gap
        # invalidates planned risk/reward, skip; this is not retroactive selection.
        entry=float(x.Open.iloc[i+1])
        last=float(x.Close.iloc[i])
        if abs(entry/last-1)>0.05:continue
        stop_pct=100*(entry-r["stop"])/entry
        take_pct=100*(r["trade_target"]/entry-1)
        if not (0<stop_pct<=12 and take_pct>0):continue
        fill=execute_path(x,i,entry,stop_pct,take_pct)
        hold=(float(x.Close.iloc[i+HORIZON])/entry-1)*100-TRADING_COST_PCT
        scenarios={m:execute_variant(x,i,entry,r["stop"],r["trade_target"],m)
                   for m in ("TAKE_STOP","HOLD_10","STOP_10","HALF_TRAIL")}
        samples.append({"as_of":str(x.index[i].date()),"signal":signal,
                        "exit":fill["exit_reason"],"net_return_pct":fill["net_return_pct"],
                        "hold_net_pct":hold,"variants":scenarios})
    direction={}
    for group in ("BUY","HOLD","SELL","AVOID"):
        vals=[o["forward_10d_net_pct"] for o in snapshots if o["group"]==group]
        direction[group]={"observations":len(vals),
                          "positive_10d":int(sum(v>0 for v in vals)),
                          "negative_10d":int(sum(v<0 for v in vals)),
                          "sum_forward_net_pct":round(float(sum(vals)),6)}
    audit={"historical_decisions":len(snapshots),"direction_counts":direction,
           "trade_samples":samples}
    if not samples:
        return {"trades":0,"reason":"유효한 매수 검토 신호가 없음",**audit}
    r=np.array([s["net_return_pct"] for s in samples])
    b=np.array([s["hold_net_pct"] for s in samples])
    return {"trades":len(samples),"mean_trade_net_pct":round(float(r.mean()),3),
            "win_rate_pct":round(float(100*np.mean(r>0)),2),
            "net_return_sum_pct":round(float(r.sum()),6),
            "hold_return_sum_pct":round(float(b.sum()),6),
            "winning_trades":int(np.sum(r>0)),
            "mean_same_entry_10d_hold_pct":round(float(b.mean()),3),
            "worst_trade_pct":round(float(r.min()),3),
            "stop_or_gap_count":sum(s["exit"] in ("stop","gap_stop") for s in samples),
            "take_count":sum(s["exit"] in ("take","gap_take") for s in samples),**audit}

def run():
    samples,universe=pick_300()
    report={"time_utc":datetime.now(timezone.utc).isoformat(),
            "status":"300_ticker_exploratory_diagnostic_not_validated",
            "requested_tickers":300,
            "sample_allocation":SAMPLE_ALLOCATION,
            "sampling":"all present-day top 100 from each market; no extra ETFs; not historical membership",
            "universe_sources":{m:universe[m+"_source"] for m in SAMPLE_ALLOCATION},
            "horizon_sessions":HORIZON,"round_trip_cost_pct":TRADING_COST_PCT,
            "assumptions":["300 present-day top-listed companies, 100 per market; survivorship bias",
                           "missing historical investor flow and true historical constituent lists",
                           "next-session open entry, 5% maximum opening gap",
                           "stop touched first if same bar hits both thresholds",
                           "actual daily open fill if gapped through stop",
                           "trades spaced 10 sessions per ticker, correlated across tickers",
                           "no price impact, benchmark or corporate actions validation"],
            "tickers":{}}
    def worker(ticker):
        market,code,name=ticker
        try:
            v=test_ticker(market,code)
        except Exception as exc:
            v={"error":str(exc)[:250]}
        return market,code,name,v
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures=[pool.submit(worker,t) for t in samples]
        for idx,future in enumerate(as_completed(futures),1):
            market,code,name,v=future.result()
            report["tickers"][market+":"+code]={"name":name,**v}
            if idx%10==0:print(f"Completed {idx}/300: {market}:{code}",flush=True)
    # Present order exactly as originally ranked, not completion order.
    report["tickers"]={
        market+":"+code:report["tickers"][market+":"+code] for market,code,_ in samples
    }
    report["analysed_tickers"]=sum("error" not in v for v in report["tickers"].values())
    report["failed_tickers"]=300-report["analysed_tickers"]
    report["tickers_with_trades"]=sum(v.get("trades",0)>0 for v in report["tickers"].values())
    report["total_trades"]=sum(v.get("trades",0) for v in report["tickers"].values())
    buckets={}
    for key,v in report["tickers"].items():
        market=key.split(":",1)[0]
        buckets.setdefault(market,{"requested":0,"analysed":0,"failed":0,"trades":0})
        buckets[market]["requested"]+=1
        buckets[market]["analysed"]+=int("error" not in v)
        buckets[market]["failed"]+=int("error" in v)
        buckets[market]["trades"]+=v.get("trades",0)
    report["per_market"]=buckets
    trades=report["total_trades"]
    report["aggregate"]={
        "trades":trades,
        "mean_trade_net_pct":round(sum(v.get("net_return_sum_pct",0) for v in report["tickers"].values())/trades,3) if trades else None,
        "mean_same_entry_hold_net_pct":round(sum(v.get("hold_return_sum_pct",0) for v in report["tickers"].values())/trades,3) if trades else None,
        "win_rate_pct":round(100*sum(v.get("winning_trades",0) for v in report["tickers"].values())/trades,2) if trades else None,
        "no_lookahead_verified":True,
        "statistical_significance_verified":False,
        "after_tax_verified":False}
    direction_summary={}
    for group in ("BUY","HOLD","SELL","AVOID"):
        direction=[v.get("direction_counts",{}).get(group,{}) for v in report["tickers"].values()]
        n=sum(d.get("observations",0) for d in direction)
        pos=sum(d.get("positive_10d",0) for d in direction)
        neg=sum(d.get("negative_10d",0) for d in direction)
        total=sum(d.get("sum_forward_net_pct",0) for d in direction)
        direction_summary[group]={"observations":n,"positive_10d":pos,
            "positive_rate_pct":round(100*pos/n,2) if n else None,
            "mean_forward_10d_net_pct":round(total/n,3) if n else None,
            "direction_match_pct":round(100*(pos if group=="BUY" else neg)/n,2) if n and group in ("BUY","SELL") else None}
    report["signal_direction_audit"]=direction_summary
    for market in SAMPLE_ALLOCATION:
        market_rows=[v for key,v in report["tickers"].items() if key.startswith(market+":")]
        market_directions={}
        for group in ("BUY","HOLD","SELL","AVOID"):
            dims=[v.get("direction_counts",{}).get(group,{}) for v in market_rows]
            n=sum(x.get("observations",0) for x in dims)
            pos=sum(x.get("positive_10d",0) for x in dims)
            neg=sum(x.get("negative_10d",0) for x in dims)
            total=sum(x.get("sum_forward_net_pct",0) for x in dims)
            market_directions[group]={"observations":n,"positive_rate_pct":round(100*pos/n,2) if n else None,
                 "direction_match_pct":round(100*(pos if group=="BUY" else neg)/n,2) if n and group in ("BUY","SELL") else None,
                 "mean_forward_10d_net_pct":round(total/n,3) if n else None}
        report["per_market"][market]["signal_direction_audit"]=market_directions
    report["historical_decision_count"]=sum(v.get("historical_decisions",0) for v in report["tickers"].values())
    # Chronological holdout: no parameter is selected using the test window.
    # The variants are frozen above; report both eras side by side.
    trades_all=[]
    for key,v in report["tickers"].items():
        for trade in v.get("trade_samples",[]):
            trades_all.append({"market":key.split(":",1)[0],"ticker":key,
                               **trade})
    if trades_all:
        dates=sorted({t["as_of"] for t in trades_all})
        split_date=dates[max(0,int(len(dates)*0.7)-1)]
        summary_by_split={}
        for subset,selected in (
            ("development",[t for t in trades_all if t["as_of"]<=split_date]),
            ("holdout",[t for t in trades_all if t["as_of"]>split_date])):
            stats={}
            for variant in ("TAKE_STOP","HOLD_10","STOP_10","HALF_TRAIL"):
                values=[t["variants"][variant] for t in selected]
                stats[variant]={"n":len(values),
                    "mean_net_return_pct":round(float(np.mean(values)),3) if values else None,
                    "win_rate_pct":round(float(100*np.mean(np.array(values)>0)),2) if values else None}
            summary_by_split[subset]=stats
        report["variant_holdout_audit"]={
            "split_date":split_date,
            "predefined_variants":["TAKE_STOP","HOLD_10","STOP_10","HALF_TRAIL"],
            "selection_warning":"Comparisons are descriptive, not optimized for test outcomes",
            "results":summary_by_split}
    else:report["variant_holdout_audit"]={"error":"no trades"}
    report["verification_complete"]=(report["analysed_tickers"]==300)
    # Non-negotiable commercial readiness gate. Evaluate ONLY the held-out dates;
    # no selection or tuning is allowed based on their observed returns.
    held=report.get("variant_holdout_audit",{}).get("results",{}).get("holdout",{})
    current=held.get("TAKE_STOP",{})
    benchmark=held.get("HOLD_10",{})
    n=current.get("n",0)
    current_mean=current.get("mean_net_return_pct")
    benchmark_mean=benchmark.get("mean_net_return_pct")
    acceptable=(report["verification_complete"] and n>=50 and
                current_mean is not None and benchmark_mean is not None and
                current_mean>0 and current_mean>benchmark_mean)
    report["release_gate"]={
        "status":"PASS" if acceptable else "FAIL",
        "holdout_trades":n,"minimum_holdout_trades":50,
        "holdout_strategy_mean_net_pct":current_mean,
        "holdout_identical_entry_hold_mean_net_pct":benchmark_mean,
        "requirements":["300/300 ticker analyses","at least 50 untouched holdout trades",
                        "positive held-out strategy mean net return",
                        "strategy return better than same-entry 10-session hold"],
        "warning":"PASS would still not validate commercial fitness: survivorship bias, missing flows and correlated samples remain"}
    return report

if __name__=="__main__":
    v=run()
    Path("data").mkdir(exist_ok=True)
    Path("data/trade-strategy-audit.json").write_text(
        json.dumps(v,ensure_ascii=False,indent=2),encoding="utf8")
    print(json.dumps({k:v[k] for k in ("requested_tickers","analysed_tickers",
                     "failed_tickers","tickers_with_trades","total_trades",
                     "per_market","verification_complete")},ensure_ascii=False,indent=2))
    if not v["verification_complete"]:
        raise SystemExit(f"300종목 전수 분석 미완료: {v['failed_tickers']}개 조회 실패")
