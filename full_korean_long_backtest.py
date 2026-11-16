"""100-stock Korean market historical lookback comparison (research only).

Fetch each symbol ONCE from 2021-01-01, construct both cohorts from identical
downloaded rows, and evaluate both on the same dates and same stock membership.
Today's top-100 universe is subject to survivorship and selection bias.
The reused Apr-Oct 2026 evaluation interval is NOT an independent holdout.
"""
import json
import os
from datetime import datetime,timezone
from pathlib import Path
from engine import compute
from providers import get_historical_prices
from universe import get_universe
from direction3 import make_direction_rows
from nonlinear_direction import run_nonlinear

HISTORY_START="2021-01-01"
RECENT_START="2024-07-20"
EVALUATION_END="2026-11-01"

def run(market):
    if market not in ("KR","KQ"):
        raise ValueError("STOCK_LONG_MARKET must be KR or KQ")
    u=get_universe()
    excluded={"KR":{"472150","0262B0"},"KQ":set()}[market]
    symbols=[a[0] for a in u[market] if a[0] not in excluded][:100]
    rows=[];errors={};per_stock={}
    report={"created_utc":datetime.now(timezone.utc).isoformat(),
        "market":market,"requested":100,"listed":len(symbols),
        "universe_source":u.get(market+"_source"),
        "price_requested_start":HISTORY_START,
        "recent_comparison_start":RECENT_START,
        "evaluation_start":"2026-04-01",
        "evaluation_end":EVALUATION_END,
        "labels_definition":"10 trading sessions >+3 UP; <-3 DOWN; else SIDEWAYS",
        "research_only":True,"live_approved":False,"independent_holdout":False,
        "limitations":[
            "Present-day top-100 stock membership introduces survivorship bias",
            "Previously inspected evaluation dates are not a fresh independent test",
            "Provider data may be shorter than the requested start",
            "Overlapping 10-session labels do not represent independent trades",
            "Prediction classification is not net trading profitability"],
        "per_ticker":per_stock,"errors":errors,"comparisons":{}}
    for i,ticker in enumerate(symbols,1):
        try:
            price,source=get_historical_prices(ticker,market,start_date=HISTORY_START)
            indicator=compute(price)
            labeled=make_direction_rows(indicator,ticker,market,end_date=EVALUATION_END)
            if len(labeled)<1:raise ValueError("no labeled outcomes")
            rows.extend(labeled)
            per_stock[ticker]={"first":str(price.index[0].date()),
                "last":str(price.index[-1].date()),"price_sessions":len(price),
                "labeled":len(labeled),"source":source}
        except Exception as exc:
            errors[ticker]=str(exc)[:180]
        if i%20==0:
            print("FULL_LONG_PROGRESS",market,i,len(per_stock),flush=True)
    report["loaded"]=len(per_stock)
    report["price_history_coverage_pct"]=round(100*len(per_stock)/100,2)
    if len(per_stock)>=90 and len(symbols)==100:
        for key,start in (("recent_850d",RECENT_START),("five_years",HISTORY_START)):
            subset=[r for r in rows if r["signal_date"]>=start]
            try:
                result=run_nonlinear(subset,eval_end=EVALUATION_END)
                report["comparisons"][key]={"status":result["status"],
                    "label_count":len(subset),
                    "train_mature":result.get("train_mature"),
                    "validation_count":result.get("validation_count"),
                    "diagnostic":result.get("diagnostic"),
                    "chosen_on_validation":result.get("chosen_on_validation"),
                    "per_ticker_diagnostic":result.get("per_ticker_diagnostic")}
            except Exception as exc:
                report["comparisons"][key]={"status":"FAILED",
                    "error":str(exc)[:300],"label_count":len(subset)}
        a=report["comparisons"].get("recent_850d",{})
        b=report["comparisons"].get("five_years",{})
        if a.get("status")==b.get("status")=="RESEARCH_ONLY":
            n1=a["diagnostic"]["samples"];n2=b["diagnostic"]["samples"]
            c1=set(a["per_ticker_diagnostic"]);c2=set(b["per_ticker_diagnostic"])
            report["paired_same_sample"]=(n1==n2 and c1==c2 and
                all(a["per_ticker_diagnostic"][t]["samples"]==
                    b["per_ticker_diagnostic"][t]["samples"] for t in c1))
            if report["paired_same_sample"]:
                report["accuracy_change_pp"]=round(
                    b["diagnostic"]["accuracy_pct"]-
                    a["diagnostic"]["accuracy_pct"],2)
                report["brier_change"]=round(
                    b["diagnostic"]["brier"]-
                    a["diagnostic"]["brier"],6)
                report["per_stock_change_pp"]={
                    ticker:round(b["per_ticker_diagnostic"][ticker]["accuracy_pct"]-
                                 a["per_ticker_diagnostic"][ticker]["accuracy_pct"],2)
                    for ticker in sorted(c1)}
            else:
                report["comparison_status"]="NOT_PAIRED_NO_ACCURACY_CLAIM"
    else:
        report["status"]="INSUFFICIENT_MARKET_COVERAGE"
    path=Path(f"data/full-long-{market}-20261116.json")
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    slim={k:{"status":v.get("status"),"samples":(v.get("diagnostic") or {}).get("samples"),
            "accuracy":(v.get("diagnostic") or {}).get("accuracy_pct"),
            "brier":(v.get("diagnostic") or {}).get("brier")}
           for k,v in report["comparisons"].items()}
    print("FULL_LONG_RESULT",json.dumps({"market":market,
           "loaded":len(per_stock),"errors":errors,"compared":slim,
           "paired":report.get("paired_same_sample"),
           "accuracy_change_pp":report.get("accuracy_change_pp"),
           "brier_change":report.get("brier_change")},ensure_ascii=False),flush=True)
    return report

if __name__=="__main__":
    run(os.environ.get("STOCK_LONG_MARKET","KR").strip().upper())
