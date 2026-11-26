"""오늘 종가 기준 B+D+E 단계별 후보 판정 (연구용).

동결된 63종목 감사(`data/bde-latest.json`)에서 단계 정의·파라미터·선택 변형을 그대로
읽어, 오늘 종가의 원신호가 A/B/B+D/C/D/B+D+E 각 단계에서 통과하는지·막히면 어떤
사유인지 기록한다. 게이트 기준은 감사 값 그대로이며 이 스크립트는 아무 기준도
낮추거나 새로 고르지 않는다.

- 시세: `providers.get_historical_prices` (FinanceDataReader → NAVER, 무료)
- 공시: `disclosures` 공개 DART 검색(무료, 키 불필요). 이 스크립트는 최근 창만
  필요하므로 2024-01-01부터 수집한다(감사는 2015-12-01부터).
- 실적: `OPENDART_API_KEY`가 있으면 OpenDART, 없으면 E단계는 판정 보류(ABSTAIN)로
  기록한다. 보류는 탈락이 아니며, 화면도 그렇게 표시한다.

사용법:
  python live_screen.py                 # 오늘 종가로 전체 실행(캐시 재사용, 재실행 가능)
  python live_screen.py --limit 12      # 한 번에 처리할 종목 수 제한(시간 제한 환경)
  python live_screen.py --status
"""
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import audit_63_fast as A
import bde_dashboard as bd
from disclosures import disclosure_features_for_signal, get_company_disclosures
from engine import compute
from market_context import build_market_context
from providers import get_historical_prices, get_market_series_bundle
from strategy_lab import PARAMS, qualifying

SUMMARY_PATH = Path("data/bde-latest.json")
OUT_PATH = Path("data/live-candidates-latest.json")
MARKET = {"KOSPI": "KR", "KOSDAQ": "KQ", "TIGER_ETF": "KR", "SOXL": "US"}
RECENT_BARS = 12
LOOKBACK = 220
DISCLOSURE_START = "2024-01-01"
KST = timezone(timedelta(hours=9))


def _log(message):
    print(message, flush=True)


def load_config(summary_path=SUMMARY_PATH):
    summary = bd.load_summary(summary_path)
    if not summary or not summary.get("cohorts"):
        raise SystemExit("B+D+E 요약이 없습니다. 먼저 `python bde_dashboard.py <audit-63-latest.json>`를 실행하세요.")
    stage_labels = {label: stage for label, stage in (summary.get("stage_labels") or {}).items() if stage}
    config = {"stage_labels": stage_labels, "stage_order": summary.get("stage_order") or list(bd.STAGE_ORDER),
              "sample_start": (summary.get("source") or {}).get("sample_start") or "2016-01-01",
              "cohorts": {}}
    for cohort, block in summary["cohorts"].items():
        config["cohorts"][cohort] = {
            "selected_label": (block.get("selected_policy") or {}).get("label"),
            "legacy_variant": block.get("legacy_variant"),
            "stage_params": {label: (entry or {}).get("params") or {}
                             for label, entry in (block.get("stage_params") or {}).items()},
            "instruments": block.get("instruments") or [],
        }
    return config


def _plan_levels(close, atr, variant):
    params = PARAMS.get(variant) or {"rr": 1.7, "stop_atr": 1.7}
    stop = max(close * 0.905, close - params["stop_atr"] * atr)
    stop = min(stop, close * 0.98)
    return stop, close + params["rr"] * (close - stop)


def _row(ticker, cohort, strategy, idx, x, frame, context, gap, market_ret60, disclosure):
    day = str(pd.Timestamp(x.index[idx]).date())
    row = x.iloc[idx]
    close = float(row.Close)
    atr = float(row.ATR)
    prior_high = float(x.High.iloc[idx - 20:idx].max())
    features = (context.get("features", {}).get(ticker, {}) or {}).get(day, {})
    base60 = float(frame.Close.iloc[max(0, idx - 60)])
    returns = frame.Close.astype(float).pct_change().iloc[max(0, idx - 20):idx + 1].dropna()
    record = disclosure.get(ticker) or {}
    available = bool((record.get("diagnostics") or {}).get("available") and record.get("records"))
    disclosed = disclosure_features_for_signal(record.get("records") or [], day, frame.index,
                                               available=available)
    market60 = market_ret60.get(day)
    stop, take = _plan_levels(close, atr, strategy)
    return {
        "ticker": ticker, "cohort": cohort, "strategy": strategy, "signal_date": day,
        "close": round(close, 2), "atr": round(atr, 2),
        "atr_pct": round(100 * atr / close, 2),
        "rsi": None if not np.isfinite(float(row.RSI)) else round(float(row.RSI), 2),
        "adx": None if not np.isfinite(float(row.ADX)) else round(float(row.ADX), 2),
        "ma200": round(float(row.MA200), 2),
        "stop": round(stop, 2), "take": round(take, 2),
        "risk_pct": round(100 * (close - stop) / close, 2),
        "rr_planned": (PARAMS.get(strategy) or {}).get("rr"),
        "signal_trend_200_pct": round(100 * (close / float(row.MA200) - 1), 2),
        "signal_volatility_20d_pct": round(100 * float(returns.std(ddof=0)), 4) if len(returns) else None,
        "relative_strength_60d_pct": (round(100 * (close / base60 - 1) - float(market60), 2)
                                      if market60 is not None and np.isfinite(market60) and base60 else None),
        "market_ma200_gap_pct": (round(float(gap[day]), 2) if day in gap and np.isfinite(gap[day]) else None),
        "market_regime": features.get("market_regime"),
        "disclosure_data_available": available,
        **disclosed,
    }


def _stage_decisions(row, stage_order, stage_labels, stage_params):
    decisions = {}
    for label in stage_order:
        stage = stage_labels.get(label)
        if not stage:
            continue
        reason = A._stage_gate_rejection(row, stage, stage_params.get(label) or {})
        decisions[label] = {"accepted": reason is None, "reason": reason}
    return decisions


CONDITION_LABELS = {
    "close_above_ma200": "종가 > MA200",
    "pdi_above_mdi": "PDI > MDI",
    "ma200_rising_20": "MA200 20일 상승",
    "ma60_above_ma200": "MA60 > MA200",
    "atr_pct_in_range": "ATR/가격 0.6~7.5%",
    "adx_ge_18": "ADX >= 18",
    "close_within_ma20_band": "종가 <= MA20×1.08",
}


def _cohort_diagnostics(frames, context, gap, bar_date):
    """원신호가 왜 0건인지 설명할 시장·조건 진단(게이트와 별개)."""
    market = (context.get("market_rows") or {}).get(bar_date) or {}
    failures = {label: 0 for label in CONDITION_LABELS.values()}
    near = []
    checked = 0
    for ticker, frame in frames.items():
        x = compute(frame)
        if len(x) < LOOKBACK + 2 or str(pd.Timestamp(x.index[-1]).date()) != bar_date:
            continue
        checked += 1
        row = x.iloc[-1]
        close, atr = float(row.Close), float(row.ATR)
        values = {
            "close_above_ma200": bool(close > float(row.MA200)),
            "pdi_above_mdi": bool(float(row.PDI) > float(row.MDI)),
            "ma200_rising_20": bool(row.MA200_UP20),
            "ma60_above_ma200": bool(float(row.MA60) > float(row.MA200)),
            "atr_pct_in_range": bool(0.006 <= atr / close <= 0.075),
            "adx_ge_18": bool(float(row.ADX) >= 18),
            "close_within_ma20_band": bool(close <= float(row.MA20) * 1.08),
        }
        for key, passed in values.items():
            if not passed:
                failures[CONDITION_LABELS[key]] += 1
        if all(values.values()):
            prior_high = float(x.High.iloc[-21:-1].max()) if len(x) > 21 else None
            near.append({"ticker": ticker, "close": round(close, 2),
                         "breakout_pct": round(100 * (close / prior_high - 1), 2) if prior_high else None,
                         "rsi": round(float(row.RSI), 1), "adx": round(float(row.ADX), 1),
                         "volume_ratio": (round(float(row.Volume) / float(row.VOL20), 2)
                                          if np.isfinite(float(row.VOL20)) and float(row.VOL20) > 0 else None)})
    return {"symbols_checked": checked,
            "market_regime": (context.get("regime_by_date") or {}).get(bar_date),
            "market_index_ma200_gap_pct": _num(market.get("market_index_ma200_gap_pct")),
            "market_volatility_ratio": _num(market.get("market_volatility_ratio")),
            "vix_expanding_percentile": _num(market.get("vix_expanding_percentile")),
            "usdkrw_20d_change_pct": _num(market.get("usdkrw_change_20d_pct")),
            "cohort_proxy_ma200_gap_pct": _num(gap.get(bar_date)),
            "condition_failures": failures,
            "near_misses": sorted(near, key=lambda r: r["ticker"])[:5]}


def _num(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return round(value, 2) if np.isfinite(value) else None


def _earnings_block(tickers, signal_days):
    """OpenDART 키가 있을 때만 실적 피처를 붙인다. 없으면 전부 보류(ABSTAIN)로 기록."""
    from opendart import earnings_features_for_signal, get_company_financials
    key_present = bool(os.environ.get("OPENDART_API_KEY", "").strip())
    features = {}
    diagnostics = {"key_present": key_present, "requested": len(tickers), "available": 0,
                   "errors": [], "mode": "opendart" if key_present else "abstain_no_api_key"}
    if not key_present:
        for ticker in tickers:
            for day in signal_days.get(ticker, []):
                features[(ticker, day)] = earnings_features_for_signal([], day, corp_code=None,
                                                                       abstain=True)
        return features, diagnostics
    for ticker in tickers:
        try:
            record = get_company_financials(ticker, start_year=2021, end_year=datetime.now(KST).year)
        except Exception as exc:
            diagnostics["errors"].append({"ticker": ticker, "error": str(exc)[:160]})
            record = {"records": [], "diagnostics": {}}
        records = record.get("records") or []
        if records:
            diagnostics["available"] += 1
        for day in signal_days.get(ticker, []):
            features[(ticker, day)] = earnings_features_for_signal(
                records, day, corp_code=(record.get("diagnostics") or {}).get("corp_code"),
                abstain=False)
    return features, diagnostics


def run(limit=None, summary_path=SUMMARY_PATH, out_path=OUT_PATH, recent_bars=RECENT_BARS):
    config = load_config(summary_path)
    cohorts = {name: block for name, block in config["cohorts"].items() if block.get("instruments")}
    tickers_by_cohort = {name: [row["ticker"] for row in block["instruments"]]
                         for name, block in cohorts.items()}
    names = {row["ticker"]: row.get("name") for block in cohorts.values() for row in block["instruments"]}
    _log("MACRO FETCH")
    macro, macro_diag = get_market_series_bundle(start_date="2016-01-01")
    frames, sources, price_errors = {}, {}, {}
    processed = 0
    for cohort, tickers in tickers_by_cohort.items():
        for ticker in tickers:
            if limit is not None and processed >= limit and ticker not in frames:
                continue
            try:
                frame, source = get_historical_prices(ticker, MARKET[cohort],
                                                      start_date=config["sample_start"],
                                                      min_bars=1 if ticker == "0262B0" else 250)
                frames[ticker] = frame
                sources[ticker] = {"source": source, "last": str(frame.index[-1].date()),
                                   "bars": int(len(frame))}
                processed += 1
                _log(f"PRICE {cohort} {ticker} {source} last={frame.index[-1].date()} bars={len(frame)}")
            except Exception as exc:
                price_errors[ticker] = str(exc)[:200]
                _log(f"PRICE_FAIL {cohort} {ticker} {exc}")
    korean = [t for c in ("KOSPI", "KOSDAQ") if c in tickers_by_cohort
              for t in tickers_by_cohort[c] if t in frames]
    disclosure, disclosure_diag = {}, {"requested": len(korean), "available": 0, "errors": []}
    for ticker in korean:
        try:
            record = get_company_disclosures(ticker, names.get(ticker) or ticker,
                                             start_date=DISCLOSURE_START)
            disclosure[ticker] = record
            if record.get("records"):
                disclosure_diag["available"] += 1
        except Exception as exc:
            disclosure_diag["errors"].append({"ticker": ticker, "error": str(exc)[:160]})
            _log(f"DISC_FAIL {ticker} {exc}")
    _log(f"DISC {json.dumps({k: v for k, v in disclosure_diag.items() if k != 'errors'}, ensure_ascii=False)}")

    signal_days = {}
    cohort_rows = {}
    for cohort, tickers in tickers_by_cohort.items():
        present = [t for t in tickers if t in frames]
        if not present:
            cohort_rows[cohort] = {"status": "NO_PRICE_DATA", "symbols": 0}
            continue
        cframes = {t: frames[t] for t in present}
        context = build_market_context(cframes, macro, cohort, {})
        breadth, gap, market_ret60 = A._make_cohort_context(cframes)
        variant = cohorts[cohort].get("legacy_variant")
        rows = []
        bar_dates = []
        for ticker, frame in cframes.items():
            x = compute(frame)
            if len(x) < LOOKBACK + 2:
                continue
            last = len(x) - 1
            bar_dates.append(str(pd.Timestamp(x.index[last]).date()))
            start = max(LOOKBACK, last - recent_bars + 1)
            days = [str(pd.Timestamp(x.index[i]).date()) for i in range(start, last + 1)]
            signal_days[ticker] = days
            for i in range(start, last + 1):
                row, prev = x.iloc[i], x.iloc[i - 1]
                if not (np.isfinite(float(row.Close)) and np.isfinite(float(row.ATR)) and float(row.ATR) > 0):
                    continue
                prior_high = float(x.High.iloc[i - 20:i].max())
                for strategy in PARAMS:
                    try:
                        if not qualifying(row, prev, MARKET[cohort], strategy, prior_high):
                            continue
                    except Exception:
                        continue
                    if variant and strategy != variant:
                        continue
                    try:
                        rows.append(_row(ticker, cohort, strategy, i, x, frame, context, gap,
                                         market_ret60, disclosure))
                    except Exception as exc:
                        _log(f"ROW_FAIL {ticker} {i} {exc}")
        cohort_rows[cohort] = {"symbols": len(cframes), "rows": rows,
                               "bar_date": max(bar_dates) if bar_dates else None,
                               "variant": variant,
                               "diagnostics": _cohort_diagnostics(cframes, context, gap,
                                                                 max(bar_dates) if bar_dates else None)}
    earnings, earnings_diag = _earnings_block(
        [t for t in frames if t in signal_days], signal_days)
    for row in (r for block in cohort_rows.values() for r in block.get("rows", [])):
        feature = earnings.get((row["ticker"], row["signal_date"])) or {}
        row.update({k: (None if isinstance(v, float) and not np.isfinite(v) else v)
                    for k, v in feature.items()})
        block = cohorts[row["cohort"]]
        row["stages"] = _stage_decisions(row, config["stage_order"], config["stage_labels"],
                                         block.get("stage_params") or {})
        row["name"] = names.get(row["ticker"])
    output = {"generated_at_kst": str(pd.Timestamp.now(tz="Asia/Seoul")),
              "status": "RESEARCH_ONLY_NOT_LIVE_APPROVED",
              "basis": {"source_summary": str(summary_path),
                        "source_report_created_utc": (bd.load_summary(summary_path) or {}).get("source", {}).get("report_created_utc"),
                        "price_source": "FinanceDataReader → NAVER via providers.get_historical_prices",
                        "disclosure_window_start": DISCLOSURE_START,
                        "sample_start": config["sample_start"],
                        "recent_bars": recent_bars,
                        "earnings": earnings_diag},
              "market_series": {k: {"status": v.get("status"), "last_date": v.get("last_date"),
                                    "source": v.get("source")} for k, v in macro_diag.items()},
              "price_sources": sources, "price_errors": price_errors,
              "disclosure": {k: v for k, v in disclosure_diag.items() if k != "errors"},
              "cohorts": {}, "notes": []}
    for cohort, block in cohort_rows.items():
        if "rows" not in block:
            output["cohorts"][cohort] = block
            continue
        rows = block["rows"]
        label = cohorts[cohort].get("selected_label")
        bar_date = block.get("bar_date")
        counts = {name: 0 for name in config["stage_order"]}
        for row in rows:
            for name, decision in row["stages"].items():
                if decision["accepted"]:
                    counts[name] = counts.get(name, 0) + 1
        today = [row for row in rows if row["signal_date"] == bar_date]
        output["cohorts"][cohort] = {
            "symbols": block["symbols"], "signal_bar_date": bar_date,
            "selected_label": label, "legacy_variant": block.get("variant"),
            "raw_signal_n": len(today),
            "counts_by_stage": counts,
            "today_signals": sorted(today, key=lambda r: (r["ticker"], r["strategy"])),
            "recent_signals": sorted(rows, key=lambda r: (r["signal_date"], r["ticker"]),
                                     reverse=True)[:20],
            "diagnostics": block.get("diagnostics"),
        }
    bar_dates = [b.get("signal_bar_date") for b in output["cohorts"].values() if b.get("signal_bar_date")]
    output["signal_bar_date"] = max(bar_dates) if bar_dates else None
    if earnings_diag["mode"] == "abstain_no_api_key":
        output["notes"].append("OpenDART 키가 없어 E(실적) 단계는 판정 보류(ABSTAIN)입니다. "
                               "보류는 통과도 탈락도 아니며, B+D+E 판정을 확정하려면 키가 있는 환경에서 다시 실행해야 합니다.")
    output["notes"].append("이 화면은 오늘 종가의 원신호에 감사 게이트를 그대로 적용한 결과이며, "
                           "게이트 기준·종목·기간 상수는 감사와 동일합니다.")
    output["notes"].append("연구용입니다. 실전 매수 승인이 아니며 주문 기능은 없습니다.")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    _log("WROTE " + str(out_path) + " " + json.dumps(
        {"bar": output["signal_bar_date"],
         "counts": {c: b.get("counts_by_stage") for c, b in output["cohorts"].items()},
         "raw": {c: b.get("raw_signal_n") for c, b in output["cohorts"].items()}}, ensure_ascii=False))
    return output


def status(path=OUT_PATH):
    if not Path(path).exists():
        print("NO_LIVE_FILE " + str(path))
        return 1
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    print(json.dumps({"generated_at_kst": data.get("generated_at_kst"),
                      "signal_bar_date": data.get("signal_bar_date"),
                      "counts": {c: b.get("counts_by_stage") for c, b in (data.get("cohorts") or {}).items()},
                      "raw_signal_n": {c: b.get("raw_signal_n") for c, b in (data.get("cohorts") or {}).items()},
                      "notes": data.get("notes")}, ensure_ascii=False, indent=1))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description="오늘 종가 B+D+E 후보 판정")
    parser.add_argument("--limit", type=int, default=None, help="이번 실행에서 가져올 종목 수 상한")
    parser.add_argument("--recent-bars", type=int, default=RECENT_BARS)
    parser.add_argument("--summary", default=str(SUMMARY_PATH))
    parser.add_argument("--out", default=str(OUT_PATH))
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args(argv)
    if args.status:
        return status(args.out)
    run(limit=args.limit, summary_path=args.summary, out_path=args.out,
        recent_bars=args.recent_bars)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
