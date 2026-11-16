"""Small real-market smoke audit of paired direction models.

This tests provider availability and research mechanics, NOT independent
certification or a 100/market backtest. Failing feeds must be reported honestly.
"""
from datetime import datetime, timezone
from pathlib import Path
import json
import pandas as pd
from engine import compute
from providers import get_prices
from direction3 import make_direction_rows
from regime_direction import score_regime_direction

# Fixed examples, not a point-in-time historical membership universe.
SAMPLES={
    "KR":("005930","000660","035420"),
    "KQ":("086520","247540","293490"),
    "US":("NVDA","AAPL","MSFT"),
}

def run():
    result={
        "created_utc":datetime.now(timezone.utc).isoformat(),
        "study_type":"small_fixed_watchlist_smoke_not_market_wide",
        "live_approved":False,"independent_holdout":False,
        "limitations":[
            "3 handpicked current tickers per market only; survivorship bias",
            "Previously inspected chronological periods; no independent holdout",
            "Forecast direction only; NOT backtested trade profitability",
            "Data-provider and historical adjusted price differences can affect results",
        ],"markets":{}}
    for market,symbols in SAMPLES.items():
        rows=[];sources={};errors={}
        for ticker in symbols:
            try:
                frame,src=get_prices(ticker,market)
                computed=compute(frame)
                sample=make_direction_rows(computed,ticker,market,
                                           end_date="2026-11-01")
                if len(sample)<110:
                    raise ValueError(f"Only {len(sample)} labeled rows")
                rows.extend(sample)
                sources[ticker]=src
            except Exception as exc:
                errors[ticker]=str(exc)[:240]
        market_result={
            "requested":len(symbols),"loaded":len(sources),
            "sources":sources,"errors":errors,"labels_total":len(rows),
            "status":"NOT_VERIFIED_INSUFFICIENT_DATA"}
        if len(sources)==len(symbols) and len(rows)>=330:
            report=score_regime_direction(rows,eval_end="2026-11-01")
            market_result["status"]="RESEARCH_ONLY_DIAGNOSTIC"
            market_result["comparison"]=report["comparison"]
            market_result["regimes"]={
                key:{"pooled":v["pooled"]["class_metrics"],
                     "regime":v["regime"]["class_metrics"]}
                for key,v in report["regimes"].items()}
            market_result["new_model_beats_pooled_on_diagnostic"]=bool(
                report["comparison"]["diagnostic"]["brier_improved"] and
                report["comparison"]["diagnostic"]["macro_f1_improved"])
        result["markets"][market]=market_result
        print("MARKET_SAMPLE_RESULT",market,
              json.dumps(market_result,ensure_ascii=False),flush=True)
    path=Path("data/regime-smoke-latest.json")
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,ensure_ascii=False,indent=2),
                    encoding="utf-8")
    return result

if __name__=="__main__":
    run()
