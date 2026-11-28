"""Local preview of the direction3 regime gate on the frozen KOSPI audit cohort.

Not part of CI. It rebuilds the same rows the 63-symbol audit uses for the KOSPI
cohort, runs the selected model kind/horizon, and prints the development,
validation and diagnostic decisions before and after the causal trend gate.
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from expanded_direction_audit import (_attach_context_features, _backtest_model,
    _freeze_volatility_labels, _make_rows, _rank_features)
from providers import get_historical_prices, get_market_series_bundle
from universe import AUDIT_63_BENCHMARK

MARKET = {"KOSPI": ("KR", "KOSPI"), "KOSDAQ": ("KQ", "KOSDAQ")}


def build(cohort="KOSPI", kind="band_regression", horizon=20):
    market = MARKET[cohort][0]
    tickers = list(AUDIT_63_BENCHMARK[cohort])
    frames = {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(get_historical_prices, ticker, market,
                               start_date="2016-01-01", min_bars=250): ticker
                   for ticker in tickers}
        for future in futures:
            ticker = futures[future]
            try:
                frame, _ = future.result()
                frames[ticker] = frame
            except Exception as exc:  # noqa: BLE001
                print(f"skip {ticker}: {exc}", file=sys.stderr)
    rows = []
    for ticker, frame in frames.items():
        rows.extend(_make_rows(frame, ticker, market, cohort))
    rows = _rank_features(rows)
    macro, _ = get_market_series_bundle(start_date="2016-01-01", end_date="2026-11-01")
    _attach_context_features(rows, {cohort: frames}, macro, {})
    _freeze_volatility_labels(rows)
    return rows, kind, horizon


def main():
    cohort = sys.argv[1] if len(sys.argv) > 1 else "KOSPI"
    kind = sys.argv[2] if len(sys.argv) > 2 else "band_regression"
    horizon = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    rows, kind, horizon = build(cohort, kind, horizon)
    dev = _backtest_model(rows, rows, kind, horizon, "development")
    gate = dev["temperature"]["regime_gate"]
    out = {"cohort": cohort, "model": kind, "horizon": horizon,
           "development_rows": dev["n"], "gate": gate,
           "gate_selection": {k: v for k, v in dev["temperature"]["regime_gate_selection"].items()
                              if k not in ("candidates", "top_candidates")},
           "periods": {}}
    for period in ("development", "validation", "diagnostic"):
        result = (dev if period == "development" else
                  _backtest_model(rows, rows, kind, horizon, period,
                                  temperature_info=dev["temperature"],
                                  payoffs=dev["class_payoffs_from_development"]))
        metrics = result.get("metrics") or {}
        decision = metrics.get("decision") or {}
        gated = metrics.get("regime_gate_decision") or {}
        out["periods"][period] = {
            "n": result.get("n"),
            "ungated_accuracy_pct": decision.get("accuracy_pct"),
            "gated_accuracy_pct": gated.get("accuracy_pct"),
            "ungated_counts": decision.get("predicted_counts"),
            "gated_counts": gated.get("predicted_counts"),
            "ungated_collapse": decision.get("collapse_status"),
            "gated_collapse": gated.get("collapse_status"),
            "brier": (metrics.get("mean_class_brier")),
            "majority_baseline_pct": (metrics.get("majority_class_baseline_accuracy_pct")),
            "overrides_n": (metrics.get("regime_gate") or {}).get("overrides_n"),
            "activated_n": (metrics.get("regime_gate") or {}).get("activated_rows_n"),
        }
    print(json.dumps(out, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
