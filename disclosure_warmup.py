"""Warm DART's free public HTML disclosure cache for Korean common shares."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from expanded_direction_audit import _ticker_set,korean_flow_targets
from disclosures import collect_company_disclosures


def run_warmup(*,time_budget_seconds=3300,workers=6,universe=None,fetcher=None,clock=None,progress_printer=print):
    universe=_ticker_set() if universe is None else universe
    targets=korean_flow_targets(universe)
    entries=[(row["ticker"],row["name"]) for row in targets]
    def progress(done,total,ticker,result):
        diag=result.get("diagnostics",{});records=result.get("records",[])
        progress_printer(f"DISCLOSURE_WARMUP ticker={done}/{total} code={ticker} rows={len(records)} requests={diag.get('requests_n',0)} status={'ok' if diag.get('available') else 'partial/unavailable'}",flush=True)
    collection=collect_company_disclosures(entries,time_budget_seconds=time_budget_seconds,workers=workers,
        fetcher=fetcher,clock=clock,progress=progress)
    by_ticker={}
    for ticker,result in collection.pop("results").items():
        diag=result.get("diagnostics",{});records=result.get("records",[])
        by_ticker[ticker]={"first_date":diag.get("first_date"),"last_date":diag.get("last_date"),
            "rows_n":len(records),"requests_n":diag.get("requests_n",0),"pages_n":diag.get("pages_n",0),
            "available":diag.get("available",False),"complete":diag.get("complete",False),
            "failures":diag.get("failures",[])}
    report={**collection,"by_ticker":by_ticker,"source":"DART public HTML detailSearch.ax",
        "requested_start":"2015-12-01","requested_end":"2026-11-13",
        "uncompleted_tickers":collection.get("not_attempted",[]),
        "partial_completion_exits_successfully":True}
    Path("data").mkdir(exist_ok=True)
    Path("data/disclosure-warmup-latest.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    progress_printer(json.dumps({k:report.get(k) for k in ("requested_symbols","attempted_symbols","available_symbols","budget_used_seconds","budget_exhausted","not_attempted","failures")},ensure_ascii=False),flush=True)
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--time-budget-seconds",type=float,default=3300)
    parser.add_argument("--workers",type=int,default=6)
    args=parser.parse_args(argv)
    run_warmup(time_budget_seconds=args.time_budget_seconds,workers=args.workers)
    return 0


if __name__=="__main__":raise SystemExit(main())
