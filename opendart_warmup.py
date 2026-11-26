"""Warm OpenDART annual/quarterly caches for the frozen 60 Korean stocks."""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from opendart import (OpenDartClient, RunRequestBudget, max_requests_per_run,
                      request_interval_seconds, warmup_workers_from_env)
from universe import AUDIT_63_BENCHMARK


def _new_client(client_factory, cache_root, request_budget):
    if cache_root is None:
        return client_factory(request_budget=request_budget)
    return client_factory(cache_root=cache_root, request_budget=request_budget)


def run_warmup(*, time_budget_seconds=2400, workers=4, cache_root=None,
               progress=print, client_factory=OpenDartClient, request_budget=None):
    """Collect independent ticker caches concurrently under one run-wide cap."""
    worker_count = max(1, min(8, int(workers)))
    request_budget = request_budget or RunRequestBudget()
    coordinator = _new_client(client_factory, cache_root, request_budget)
    started = time.monotonic()
    deadline = started + max(0.0, float(time_budget_seconds))
    coordinator.set_time_budget(max(0.0, deadline - time.monotonic()))
    tickers = list(AUDIT_63_BENCHMARK["KOSPI"] + AUDIT_63_BENCHMARK["KOSDAQ"])
    mapping = coordinator.get_corp_codes() if coordinator.key_present else {}
    per_ticker = {}

    def collect(index, ticker):
        remaining = max(0.0, deadline - time.monotonic())
        corp_code = mapping.get(str(ticker).zfill(6))
        if not coordinator.key_present:
            return index, ticker, {"corp_code": corp_code, "status": "ABSTAIN_NO_API_KEY",
                "rows_n": 0, "requests_n": 0, "first_rcept_date": None,
                "last_rcept_date": None, "failures": [], "available": False,
                "attempted": False, "cap_reached": False}
        if request_budget.cap_reached:
            return index, ticker, {"corp_code": corp_code, "status": "SKIPPED_REQUEST_CAP",
                "rows_n": 0, "requests_n": 0, "first_rcept_date": None,
                "last_rcept_date": None, "failures": [], "available": False,
                "attempted": False, "cap_reached": True}
        if remaining <= 0:
            return index, ticker, {"corp_code": corp_code, "status": "SKIPPED_BUDGET",
                "rows_n": 0, "requests_n": 0, "first_rcept_date": None,
                "last_rcept_date": None, "failures": [], "available": False,
                "attempted": False, "cap_reached": request_budget.cap_reached}
        if not corp_code:
            return index, ticker, {"corp_code": None, "status": "NO_CORP_CODE",
                "rows_n": 0, "requests_n": 0, "first_rcept_date": None,
                "last_rcept_date": None, "failures": [], "available": False,
                "attempted": False, "cap_reached": request_budget.cap_reached}

        client = _new_client(client_factory, cache_root, request_budget)
        client.set_time_budget(remaining)
        before = client._requests_n
        try:
            result = client.company_financials(ticker, corp_code=corp_code,
                start_year=2015, end_year=2026)
        except Exception as exc:
            result = {"records": [], "diagnostics": {"status": "COLLECTION_ERROR",
                "failures": [type(exc).__name__], "available": False,
                "attempted": True, "cap_reached": request_budget.cap_reached}}
        diagnostics = result.get("diagnostics", {})
        records = result.get("records", [])
        return index, ticker, {"corp_code": corp_code,
            "status": diagnostics.get("status"), "rows_n": len(records),
            "requests_n": client._requests_n - before,
            "first_rcept_date": diagnostics.get("first_rcept_date"),
            "last_rcept_date": diagnostics.get("last_rcept_date"),
            "failures": diagnostics.get("failures", []),
            "available": bool(diagnostics.get("available") and records),
            "attempted": bool(diagnostics.get("attempted")),
            "cap_reached": bool(diagnostics.get("cap_reached") or request_budget.cap_reached)}

    completed = 0
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [executor.submit(collect, index, ticker)
                   for index, ticker in enumerate(tickers, 1)]
        for future in as_completed(futures):
            index, ticker, summary = future.result()
            per_ticker[ticker] = summary
            completed += 1
            progress(f"OPENDART_WARMUP ticker={completed}/{len(tickers)} code={ticker} "
                     f"rows={summary['rows_n']} requests={summary['requests_n']} "
                     f"status={summary['status']}", flush=True)

    missing = [ticker for ticker in tickers if not per_ticker[ticker]["available"]]
    attempted = sum(int(row["attempted"]) for row in per_ticker.values())
    available = sum(int(row["available"]) for row in per_ticker.values())
    report = {
        "source": "OpenDART fnlttSinglAcntAll.json",
        "requested_symbols_n": len(tickers), "attempted_symbols_n": attempted,
        "available_symbols_n": available, "missing_tickers": missing,
        "key_present": coordinator.key_present,
        "network_requests_n": request_budget.requests_n,
        "requests_n": request_budget.requests_n,
        "request_limit": request_budget.limit,
        "cap_reached": request_budget.cap_reached,
        "request_interval_seconds": request_interval_seconds(),
        "workers": worker_count,
        "budget_seconds": float(time_budget_seconds),
        "budget_used_seconds": round(time.monotonic() - started, 3),
        "budget_exhausted": bool(coordinator.key_present and time.monotonic() >= deadline),
        "corp_code_symbols_n": len(mapping), "by_ticker": per_ticker,
        "cache_path": str(coordinator.cache_root),
    }
    Path("data").mkdir(exist_ok=True)
    Path("data/opendart-warmup-latest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    progress(json.dumps({key: report[key] for key in (
        "requested_symbols_n", "attempted_symbols_n", "available_symbols_n",
        "requests_n", "request_limit", "cap_reached", "workers",
        "budget_used_seconds", "budget_exhausted", "missing_tickers")},
        ensure_ascii=False), flush=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--time-budget-seconds", type=float, default=2400)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)
    run_warmup(time_budget_seconds=args.time_budget_seconds,
        workers=args.workers or warmup_workers_from_env())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
