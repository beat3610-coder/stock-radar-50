"""Long-history research: comparable out-of-period outcomes across lookback windows.

Research-only. Nine fixed symbols are neither the full universe nor an unbiased
random sample. A provider's first returned date is explicitly recorded.
"""
import json
from datetime import datetime,timezone
from pathlib import Path
from engine import compute
from providers import get_historical_prices
from direction3 import make_direction_rows
from nonlinear_direction import run_nonlinear

SYMBOLS={"KR":("005930","000660","005380"),
         "KQ":("086520","247540","035900"),
         "US":("AAPL","MSFT","NVDA")}
LOOKBACK_STARTS={"recent_850d":"2024-07-20","five_years":"2021-01-01",
                 "ten_years":"2016-01-01","fifteen_years":"2011-01-01"}
EVAL_END="2026-11-01"

def run():
    out={"created_utc":datetime.now(timezone.utc).isoformat(),
         "research_only":True,"independent_holdout":False,"live_approved":False,
         "test_start":"2026-04-01","test_end":EVAL_END,
         "samples":"3 fixed symbols per market, not 300-stock backtest",
         "warnings":["Fixed today-known stocks: survivorship bias",
                     "Historical price availability differs by provider",
                     "Recently inspected diagnostic interval is NOT an independent future test",
                     "Overlapping forward labels are not independent trades"],
         "markets":{}}
    for market,symbols in SYMBOLS.items():
        data={};failures={}
        for ticker in symbols:
            try:
                price,src=get_historical_prices(ticker,market,start_date="2011-01-01")
                x=compute(price)
                history=make_direction_rows(x,ticker,market,end_date=EVAL_END)
                data[ticker]={"rows":history,"date_first":str(price.index[0].date()),
                              "date_last":str(price.index[-1].date()),
                              "bars":len(price),"source":src}
            except Exception as exc:
                failures[ticker]=str(exc)[:220]
        result={"requested":len(symbols),"loaded":len(data),"errors":failures,
                "stocks":{ticker:{k:v for k,v in info.items() if k!="rows"}|
                          {"labeled_events":len(info["rows"])}
                          for ticker,info in data.items()},
                "windows":{}}
        for label,start in LOOKBACK_STARTS.items():
            subset=[r for ticker,info in data.items()
                    for r in info["rows"] if r["signal_date"]>=start]
            if len(data)<len(symbols) or len(subset)<300:
                result["windows"][label]={"status":"INCOMPLETE","labels":len(subset)}
                continue
            try:
                score=run_nonlinear(subset,eval_end=EVAL_END)
                result["windows"][label]={
                    "status":score["status"],"labels":len(subset),
                    "chosen_on_validation":score.get("chosen_on_validation"),
                    "diagnostic":score.get("diagnostic"),
                    "training_samples":score.get("train_mature"),
                    "evaluation_samples":score.get("diagnostic_count")}
            except Exception as exc:
                result["windows"][label]={"status":"ERROR","error":str(exc)[:240],
                                           "labels":len(subset)}
        out["markets"][market]=result
        print("LONG_HISTORY_MARKET",market,json.dumps(result,ensure_ascii=False),flush=True)
        p=Path("data/long-history-comparison.json")
        p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding="utf-8")
    return out

if __name__=="__main__":
    run()
