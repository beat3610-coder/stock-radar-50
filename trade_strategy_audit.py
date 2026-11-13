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

# The 100 candidates are selected *before* any trade outcomes are computed.
SAMPLE_ALLOCATION={"KR":34,"KQ":33,"US":33}
MAX_WORKERS=4

def pick_100():
    universe=get_universe()
    selected=[]
    for market,n in SAMPLE_ALLOCATION.items():
        # Exclude the optional leveraged ETFs; sample ordinary companies only.
        stocks=[(market,code,name) for code,name,_,sector in universe[market]
                if code not in ("472150","0262B0","SOXL")]
        if len(stocks)<n:
            raise RuntimeError(f"{market}: 필요 {n}종목, 조회된 종목 {len(stocks)}종목")
        # Spread ranks evenly across each market's top 100 rather than taking
        # only the mega-caps. Choice is deterministic and independent of returns.
        positions=np.linspace(0,min(99,len(stocks)-1),num=n,dtype=int)
        selected.extend(stocks[int(i)] for i in positions)
    assert len(selected)==100 and len({(m,c) for m,c,_ in selected})==100
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

def test_ticker(market, code):
    df,_=get_prices(code,market)
    x=compute(df)
    samples=[]
    for i in range(260,len(x)-HORIZON,STRIDE):
        data=x.iloc[:i+1]
        r=evaluate(data,market=market,flow=None)
        signal=r["signal"]
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
        samples.append({"as_of":str(x.index[i].date()),"signal":signal,
                        "exit":fill["exit_reason"],"net_return_pct":fill["net_return_pct"],
                        "hold_net_pct":hold})
    if not samples:
        return {"trades":0,"reason":"유효한 매수 검토 신호가 없음"}
    r=np.array([s["net_return_pct"] for s in samples])
    b=np.array([s["hold_net_pct"] for s in samples])
    return {"trades":len(samples),"mean_trade_net_pct":round(float(r.mean()),3),
            "win_rate_pct":round(float(100*np.mean(r>0)),2),
            "mean_same_entry_10d_hold_pct":round(float(b.mean()),3),
            "worst_trade_pct":round(float(r.min()),3),
            "stop_or_gap_count":sum(s["exit"] in ("stop","gap_stop") for s in samples),
            "take_count":sum(s["exit"] in ("take","gap_take") for s in samples)}

def run():
    samples,universe=pick_100()
    report={"time_utc":datetime.now(timezone.utc).isoformat(),
            "status":"100_ticker_exploratory_diagnostic_not_validated",
            "requested_tickers":100,
            "sample_allocation":SAMPLE_ALLOCATION,
            "sampling":"rank-stratified across present-day top 100 in each market; no ETFs; not historical membership",
            "universe_sources":{m:universe[m+"_source"] for m in SAMPLE_ALLOCATION},
            "horizon_sessions":HORIZON,"round_trip_cost_pct":TRADING_COST_PCT,
            "assumptions":["100 currently selected cap-ranked stocks; strong survivorship bias",
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
            if idx%10==0:print(f"Completed {idx}/100: {market}:{code}",flush=True)
    # Present order exactly as originally ranked, not completion order.
    report["tickers"]={
        market+":"+code:report["tickers"][market+":"+code] for market,code,_ in samples
    }
    report["analysed_tickers"]=sum("error" not in v for v in report["tickers"].values())
    report["failed_tickers"]=100-report["analysed_tickers"]
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
    report["verification_complete"]=(report["analysed_tickers"]==100)
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
        raise SystemExit(f"100종목 전수 분석 미완료: {v['failed_tickers']}개 조회 실패")
