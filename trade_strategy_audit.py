"""Next-open, stop-first, cost-aware diagnostic of trade-plan outcomes.

Selected current tickers = survivorship bias. No historical investor flows.
Do not treat this as a production-quality investment backtest.
"""
import json
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from engine import compute,evaluate
from providers import get_prices
from accuracy_audit import SAMPLES

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
    report={"time_utc":datetime.now(timezone.utc).isoformat(),
            "status":"limited_diagnostic_not_proven","horizon_sessions":HORIZON,
            "round_trip_cost_pct":TRADING_COST_PCT,
            "assumptions":["7 currently chosen tickers; survivorship bias",
                           "missing historical investor flow",
                           "next-session open entry, 5% maximum opening gap",
                           "stop touched first if same bar hits both thresholds",
                           "actual daily open fill if gapped through stop",
                           "separate trades 10 trading sessions apart",
                           "no price impact, benchmark or corporate actions validation"],
            "tickers":{}}
    for market,code in SAMPLES:
        try:report["tickers"][market+":"+code]=test_ticker(market,code)
        except Exception as e:report["tickers"][market+":"+code]={"error":str(e)[:250]}
    report["total_trades"]=sum(v.get("trades",0) for v in report["tickers"].values())
    return report

if __name__=="__main__":
    v=run()
    Path("data").mkdir(exist_ok=True)
    Path("data/trade-strategy-audit.json").write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding="utf8")
    print(json.dumps(v,ensure_ascii=False,indent=2))
