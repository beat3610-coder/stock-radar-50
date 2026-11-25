"""Warm OpenDART annual/quarterly caches for the frozen 60 Korean stocks."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from opendart import OpenDartClient, request_interval_seconds
from universe import AUDIT_63_BENCHMARK


def run_warmup(*, time_budget_seconds=1500, cache_root=None, progress=print):
    client = OpenDartClient() if cache_root is None else OpenDartClient(cache_root=cache_root)
    client.set_time_budget(time_budget_seconds)
    started = time.monotonic()
    tickers = list(AUDIT_63_BENCHMARK["KOSPI"] + AUDIT_63_BENCHMARK["KOSDAQ"])
    mapping = client.get_corp_codes() if client.key_present else {}
    per_ticker = {}
    attempted = available = 0
    for index, ticker in enumerate(tickers, 1):
        if not client.key_present or not client._budget_available():
            break
        corp_code = mapping.get(str(ticker).zfill(6))
        before = client._requests_n
        result = client.company_financials(ticker, corp_code=corp_code,
            start_year=2015, end_year=2026)
        diagnostics = result.get("diagnostics", {})
        records = result.get("records", [])
        attempted += int(diagnostics.get("attempted", False))
        available += int(diagnostics.get("available", False))
        per_ticker[ticker] = {
            "corp_code": corp_code, "status": diagnostics.get("status"),
            "rows_n": len(records), "requests_n": client._requests_n - before,
            "first_rcept_date": diagnostics.get("first_rcept_date"),
            "last_rcept_date": diagnostics.get("last_rcept_date"),
            "failures": diagnostics.get("failures", []),
        }
        progress(f"OPENDART_WARMUP ticker={index}/{len(tickers)} code={ticker} "
                 f"rows={len(records)} requests={client._requests_n-before} "
                 f"status={diagnostics.get('status')}", flush=True)
    missing = [ticker for ticker in tickers if ticker not in per_ticker]
    report = {
        "source": "OpenDART fnlttSinglAcntAll.json",
        "requested_symbols_n": len(tickers), "attempted_symbols_n": attempted,
        "available_symbols_n": available, "missing_tickers": missing,
        "key_present": client.key_present, "network_requests_n": client._requests_n,
        "request_interval_seconds": request_interval_seconds(),
        "budget_seconds": float(time_budget_seconds),
        "budget_used_seconds": round(time.monotonic() - started, 3),
        "budget_exhausted": bool(missing and client.key_present and not client._budget_available()),
        "corp_code_symbols_n": len(mapping), "by_ticker": per_ticker,
        "cache_path": str(client.cache_root),
    }
    Path("data").mkdir(exist_ok=True)
    Path("data/opendart-warmup-latest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    progress(json.dumps({key: report[key] for key in (
        "requested_symbols_n", "attempted_symbols_n", "available_symbols_n",
        "network_requests_n", "budget_used_seconds", "budget_exhausted", "missing_tickers")},
        ensure_ascii=False), flush=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--time-budget-seconds", type=float, default=1500)
    args = parser.parse_args(argv)
    run_warmup(time_budget_seconds=args.time_budget_seconds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
