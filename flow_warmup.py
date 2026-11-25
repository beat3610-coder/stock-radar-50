"""Manually warm the cached Naver historical flow pages for Korean equities."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from expanded_direction_audit import (
    MIN_END_DATE,
    _collect_extended_pool_flows,
    _ticker_set,
    korean_flow_targets,
)


def run_warmup(*,time_budget_seconds=3300,workers=16,universe=None,
               flow_fetcher=None,clock=None,progress_printer=print):
    universe=_ticker_set() if universe is None else universe
    flow_frames={}

    def progress(item,total,completed,rows,requests,status):
        if progress_printer:
            progress_printer(
                f"FLOW_WARMUP ticker={completed}/{total} code={item['ticker']} "
                f"rows={rows} requests={requests} status={status}",flush=True)

    result=_collect_extended_pool_flows(universe,flow_frames,
        budget_seconds=time_budget_seconds,workers=workers,
        flow_fetcher=flow_fetcher,progress=progress,clock=clock)
    target_by_code={x["ticker"]:x for x in korean_flow_targets(universe)}
    symbols={}
    for code,meta in result["by_ticker"].items():
        target=target_by_code.get(code,{})
        symbols[code]={"cohort":target.get("cohort"),"market":target.get("market"),
            "status":meta.get("status"),"first_date":meta.get("first_date"),
            "last_date":meta.get("last_date"),"rows_n":meta.get("rows_n",0),
            "requests_n":meta.get("requests_n",0),"attempted":meta.get("attempted",False),
            "available":meta.get("available",False),
            "failure_reason":meta.get("reason"),
            "failures":meta.get("failures",[])}
    report={**result,"requested_end":MIN_END_DATE,"source":"Naver mobile stock trend API",
        "symbols":symbols,
        "uncompleted_tickers":[x["ticker"] for x in result.get("not_attempted",[])],
        "failure_tickers":[x["ticker"] for x in result.get("failures",[])],
        "exit_policy":"Partial completion is reported and exits successfully; cached pages allow the next dispatch to resume."}
    Path("data").mkdir(exist_ok=True)
    Path("data/flow-warmup-latest.json").write_text(
        json.dumps(report,ensure_ascii=False,indent=2,default=str),encoding="utf-8")
    progress_printer(json.dumps({k:report[k] for k in (
        "requested_symbols","attempted_symbols","available_symbols","total_requests_n",
        "rows_n","budget_used_seconds","budget_exhausted","coverage_by_cohort",
        "failure_tickers","uncompleted_tickers")},ensure_ascii=False),flush=True)
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--time-budget-seconds",type=float,default=3300)
    parser.add_argument("--workers",type=int,default=16)
    args=parser.parse_args(argv)
    run_warmup(time_budget_seconds=args.time_budget_seconds,workers=args.workers)
    # Partial coverage is expected when the time budget expires; it is recorded
    # explicitly and can be resumed from the page cache in the next dispatch.
    return 0


if __name__=="__main__":
    raise SystemExit(main())
