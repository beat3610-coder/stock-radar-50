"""Independent (non-official) quote comparison and forward-only historical signal audit.

This diagnostic is not a validated trading strategy, a KRX official close check,
or a full market-universe survivorship-free backtest.
"""
import json
import math
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
from engine import compute, evaluate, normalize
from providers import get_prices

SAMPLES=[("KR","005930"),("KR","000660"),("KQ","196170"),
         ("KQ","086520"),("US","NVDA"),("US","MSFT"),("US","SOXL")]
HORIZON=10
TOLERANCE_PERCENT=0.5

def secondary_yahoo(code, market):
    import yfinance as yf
    ticker=code+(".KQ" if market=="KQ" else ".KS") if market!="US" else code
    data=yf.download(ticker,period="2y",auto_adjust=False,
                     interval="1d",progress=False,threads=False,timeout=15)
    return normalize(data),ticker

def compare_closes(left, right):
    # Compare only the most recent identical trading day; no mixing one day's
    # after-close data with another day's snapshot.
    a=left[["Close"]].copy()
    b=right[["Close"]].copy()
    a.index=pd.to_datetime(a.index).tz_localize(None).normalize() if a.index.tz is None else pd.to_datetime(a.index).tz_convert(None).normalize()
    b.index=pd.to_datetime(b.index).tz_localize(None).normalize() if b.index.tz is None else pd.to_datetime(b.index).tz_convert(None).normalize()
    overlap=a.index.intersection(b.index)
    if len(overlap)==0:
        return {"status":"unverified","reason":"No common trading date"}
    date=overlap.max()
    ca=float(a.loc[date,"Close"])
    cb=float(b.loc[date,"Close"])
    delta=abs(ca/cb-1)*100
    recent_a=a.index.max()
    recent_b=b.index.max()
    # Don't compare old, stale matches if today's available date differs.
    if recent_a!=recent_b or recent_a!=date:
        return {"status":"unverified","reason":"Latest trading dates differ",
                "primary_latest":str(recent_a.date()),"secondary_latest":str(recent_b.date()),
                "common_date":str(date.date())}
    return {"status":"agree_within_tolerance" if delta<=TOLERANCE_PERCENT else "mismatch",
            "date":str(date.date()),"primary_close":ca,
            "secondary_close":cb,"relative_difference_pct":round(delta,4),
            "tolerance_pct":TOLERANCE_PERCENT}

def historical_test(raw, market, sample_stride=7):
    # Today's selected market leaders are a survivorship-biased sample.
    # For every historical decision, use only bars available *through* t.
    x=compute(raw)
    out=[]
    for t in range(260,len(x)-HORIZON,sample_stride):
        past=x.iloc[:t+1]
        r=evaluate(past,market=market,flow=None)
        close=float(x.Close.iloc[t])
        future=float(x.Close.iloc[t+HORIZON])
        gain=(future/close-1)*100
        sig=r["signal"]
        category=("technical_candidate" if sig=="기술매수 후보·수급 미확인"
                  else "target_hold" if sig=="기술상승·매수 유보(수급 미확인)"
                  else "other")
        out.append({"as_of":str(x.index[t].date()),"category":category,
                    "return_10d_pct":gain,"score":r["score"],
                    "target":r["integrated_target"]})
    return out

def summarize(sample_rows):
    groups={}
    for row in sample_rows:
        groups.setdefault(row["category"],[]).append(row["return_10d_pct"])
    result={}
    for category,values in groups.items():
        result[category]={"observations":len(values),
                          "mean_return_10d_pct":round(sum(values)/len(values),3),
                          "win_rate_10d_pct":round(100*sum(v>0 for v in values)/len(values),2)}
    return result

def run():
    results={"as_of_utc":datetime.now(timezone.utc).isoformat(),
             "status":"diagnostic_only",
             "method":"Yahoo close vs first available primary public feed; exact trading-date match; <=0.5% tolerance",
             "backtest":{"method":"walk-forward historical snapshots, 10-session close-to-close",
                         "scope":"7 selected currently popular tickers only; market-wide inference is invalid",
                         "no_historical_investor_flows":True,
                         "transaction_costs_included":False,
                         "survivorship_bias":True,
                         "results_by_ticker":{}},
             "quote_comparisons":[]}
    all_rows=[]
    for market,code in SAMPLES:
        item={"market":market,"ticker":code}
        try:
            prices,source=get_prices(code,market)
            item["primary_source"]=source
            try:
                other,symbol=secondary_yahoo(code,market)
                item["independent_source"]="Yahoo "+symbol
                item["comparison"]=compare_closes(prices,other)
                if source.startswith("Yahoo"):
                    item["comparison"]={"status":"unverified","reason":"Primary already Yahoo; not independent"}
            except Exception as e:
                item["comparison"]={"status":"unverified","reason":"Secondary fetch failed: "+str(e)[:150]}
            try:
                rows=historical_test(prices,market)
                results["backtest"]["results_by_ticker"][market+":"+code]={
                    "observations":len(rows),"by_signal":summarize(rows)}
                all_rows.extend(rows)
            except Exception as e:
                results["backtest"]["results_by_ticker"][market+":"+code]={"error":str(e)[:180]}
        except Exception as e:
            item["comparison"]={"status":"unverified","reason":"Primary fetch failed: "+str(e)[:160]}
        results["quote_comparisons"].append(item)
    results["backtest"]["pooled_selected_sample"]=summarize(all_rows)
    results["backtest"]["total_observations"]=len(all_rows)
    return results

if __name__=="__main__":
    result=run()
    Path("data").mkdir(exist_ok=True)
    Path("data/accuracy-audit.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf8")
    print(json.dumps(result,ensure_ascii=False))
