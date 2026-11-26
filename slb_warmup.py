"""Fetch one recent 60-stock securities-lending snapshot within the 100-call cap."""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from slb import DEFAULT_CACHE_ROOT, collect_slb_snapshot
from universe import AUDIT_63_BENCHMARK


def run_warmup(*, std_dt=None, cache_root=DEFAULT_CACHE_ROOT, progress=print):
    tickers = list(AUDIT_63_BENCHMARK["KOSPI"] + AUDIT_63_BENCHMARK["KOSDAQ"])
    day = std_dt or datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y%m%d")
    snapshot = collect_slb_snapshot(tickers, day, cache_root=cache_root)
    per_ticker = {}
    for ticker in tickers:
        item = snapshot.get("dealing", {}).get(ticker, {"status": snapshot.get("status"), "records": []})
        rows = item.get("records", [])
        dates = [str(row.get("stdDt")) for row in rows if row.get("stdDt")]
        per_ticker[ticker] = {
            "isin": item.get("isin"), "status": item.get("status"),
            "rows_n": len(rows), "first_date": min(dates) if dates else None,
            "last_date": max(dates) if dates else None,
        }
        progress(f"SLB_WARMUP ticker={len(per_ticker)}/{len(tickers)} code={ticker} "
                 f"rows={len(rows)} status={item.get('status')}", flush=True)
    rank = snapshot.get("rank") or {}
    report = {
        "source": "data.go.kr B552481/SlbSvc XML",
        "requested_tickers_n": len(tickers),
        "network_requests_n": snapshot.get("requests_n", 0),
        "traffic_limit_per_day": 100,
        "expected_calls_if_uncached": 61,
        "rank_std_dt": rank.get("std_dt", day),
        "rank_rows_n": len(rank.get("records", [])),
        "status": snapshot.get("status"),
        "key_status": "present" if snapshot.get("status") != "ABSTAIN_NO_API_KEY" else "absent",
        "available_range": "recent 5 trading days from dealing endpoint; rank is one requested date",
        "backtest_status": "NOT_EVALUATED",
        "backtest_reason": "The dealing endpoint only returns recent five trading days; rank is top-100 only and consumes one of 100 daily calls.",
        "by_ticker": per_ticker,
        "slb_errors": {ticker: item.get("error") for ticker, item in snapshot.get("dealing", {}).items()
                        if item.get("error")},
    }
    Path("data").mkdir(exist_ok=True)
    Path("data/slb-warmup-latest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    progress(json.dumps({key: report[key] for key in (
        "requested_tickers_n", "network_requests_n", "traffic_limit_per_day",
        "rank_rows_n", "status", "backtest_status")}, ensure_ascii=False), flush=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=None, help="Rank as-of date YYYYMMDD; defaults to Korea today.")
    args = parser.parse_args(argv)
    run_warmup(std_dt=args.date)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
