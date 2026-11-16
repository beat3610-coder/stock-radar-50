"""Full 300-current-constituent, paired causal direction backtest.

Historical *research* only: re-used windows and present-day membership
do not constitute an independent forward test or trading approval.
"""
import json
import os
from datetime import datetime,timezone
from pathlib import Path
from engine import compute
from providers import get_prices
from universe import get_universe
from direction3 import make_direction_rows
from regime_direction import score_regime_direction

def run():
    report={"created_utc":datetime.now(timezone.utc).isoformat(),
       "study_type":"300_current_constituents_paired_direction_research",
       "research_only":True,"independent_holdout":False,"live_approved":False,
       "market_count_target":3,"symbols_per_market_target":100,
       "train_end":"2025-11-01","validation_end":"2026-04-01",
       "evaluation_end":"2026-11-01",
       "limitations":["Current-constituent survivorship bias",
                      "Previously inspected evaluation periods",
                      "Directional accuracy is not a trading profit metric",
                      "Historical prices and corporate actions depend on data provider"],
       "markets":{}}
    universe=get_universe()
    for market in ("KR","KQ","US"):
        excluded={"KR":{"472150","0262B0"},"KQ":set(),"US":{"SOXL"}}[market]
        symbols=[s[0] for s in universe[market] if s[0] not in excluded][:100]
        rows=[];errors=[]
        for i,ticker in enumerate(symbols,1):
            try:
                frame,source=get_prices(ticker,market)
                computed=compute(frame)
                labeled=make_direction_rows(computed,ticker,market,end_date="2026-11-01")
                if not labeled:
                    raise ValueError("No labeled historical rows")
                rows.extend(labeled)
            except Exception as exc:
                errors.append({"ticker":ticker,"error":str(exc)[:180]})
            if i%20==0:
                print("BACKTEST300_PROGRESS",market,i,"loaded",i-len(errors),flush=True)
        outcome={"requested":100,"symbols_listed":len(symbols),
                 "loaded":len(symbols)-len(errors),"errors":errors[:20],
                 "label_count":len(rows),"status":"INCOMPLETE"}
        if len(symbols)==100 and len(rows)>=300 and len(symbols)-len(errors)>=90:
            comparison=score_regime_direction(rows,eval_end="2026-11-01")
            outcome["status"]=("HISTORICAL_RESEARCH_ONLY" if not errors else "PARTIAL_COVERAGE_RESEARCH_ONLY")
            outcome["coverage_pct"]=round(100*outcome["loaded"]/outcome["requested"],2)
            outcome["missing_tickers_excluded_not_imputed"]=True
            outcome["comparison"]=comparison["comparison"]
            outcome["selected_on_validation"]=comparison["selected_on_validation"]
            outcome["selected_diagnostic"]=comparison["selected_diagnostic"]
            outcome["direction_diagnostic"]={
                model:{"accuracy_pct":comparison["summary"][model]["diagnostic"]["accuracy_pct"],
                       "brier":comparison["summary"][model]["diagnostic"]["multiclass_brier"],
                       "prior_brier":comparison["summary"][model]["diagnostic"]["prior_brier"],
                       "class_metrics":comparison["summary"][model]["diagnostic"]["class_metrics"],
                       "predicted":comparison["summary"][model]["diagnostic"]["predicted"],
                       "actual":comparison["summary"][model]["diagnostic"]["actual"],
                       "samples":comparison["summary"][model]["diagnostic"]["total"]}
                for model in ("pooled","regime")}
        report["markets"][market]=outcome
        print("BACKTEST300_MARKET",market,json.dumps(outcome,ensure_ascii=False),flush=True)
        p=Path("data/backtest-300-direction-20261116.json")
        p.parent.mkdir(exist_ok=True)
        p.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    return report

if __name__=="__main__":
    run()
