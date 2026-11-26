"""B+D+E 최근 필터와 오늘의 실전 후보를 앱이 읽을 수 있는 형태로 만든다.

B+D+E = RISK_OFF 하드게이트(B) + DART 공시 필터(D) + OpenDART 실적 필터(E).
전체 63종목 감사 보고서(`data/audit-63-latest.json`)는 7MB를 넘고 CI 아티팩트로만
존재한다. 이 모듈은 화면에 필요한 필드만 추려 `data/bde-latest.json`(수십 KB)으로
남기고, `app.py`와 `direction_audit.py`가 그 요약을 읽게 한다. 값은 자르지 않고
그대로 옮기며, 없는 필드는 추정하지 않고 None으로 남긴다.

사용법:
  python bde_dashboard.py data/audit-63-latest.json --run-id 45635628647 \
      --head c99238c --branch feature/opendart-earnings-20261125
  python bde_dashboard.py --status          # 커밋된 요약의 기준 시각·경과일만 확인
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_SUMMARY_PATH = Path("data/bde-latest.json")
DEFAULT_AUDIT_PATH = Path("data/audit-63-latest.json")
SCHEMA_VERSION = 1
STALE_AFTER_DAYS = 7

# 감사 게이트가 돌려주는 사유 코드를 화면 문장으로 옮긴다. 코드 자체는 보존한다.
REASON_LABELS = {
    "MARKET_REGIME_RISK_OFF": "시장 위험국면(RISK_OFF) — 신규 진입 차단",
    "PERIODIC_REPORT_TOO_RECENT": "정기보고서 제출 후 대기 거래일 미달",
    "BAD_TITLE_WITHIN_60_DAYS": "60거래일 내 악성 공시",
    "MISSING_DISCLOSURE_FEATURES": "공시 데이터 없음",
    "MISSING_TREND_FEATURES": "추세 지표 누락",
    "MARKET_MA200_GAP_BELOW_FLOOR": "시장 200일선 갭 하한 미달",
    "STOCK_TREND_200_NOT_POSITIVE": "종목 200일선 추세 음수",
    "RELATIVE_STRENGTH_60D_NOT_POSITIVE": "60일 상대강도 음수",
    "VOLATILITY_ABOVE_CAP": "변동성 상한 초과",
    "EARNINGS_THRESHOLD_NOT_MET": "실적 기준 미달",
    "OVERLAP_WITH_AN_EARLIER_ACCEPTED_TRADE": "앞선 채택 거래와 보유기간 중복",
    "STAGE_NOT_EVALUATED": "단계 미평가",
}

PERIOD_LABELS = {"development": "개발", "validation": "검증", "diagnostic": "진단"}
PERIOD_FIELDS = ("trades", "win_rate_pct", "mean_net_pct", "profit_factor",
                 "max_drawdown_pct", "max_drawdown_sequence_pct", "coverage_pct",
                 "traded_symbols", "requested_symbols", "small_sample")
COHORT_LABELS = {"KOSPI": "코스피", "KOSDAQ": "코스닥", "TIGER_ETF": "TIGER ETF", "SOXL": "SOXL"}
DIRECTION_MARKETS = {"KOSPI": "코스피", "KOSDAQ": "코스닥", "US": "미국"}
STAGE_ORDER = ("A", "B", "B+D", "C", "D", "B+D+E")


def _now_utc():
    return datetime.now(timezone.utc)


def _round(value, digits=4):
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (int, float)):
        return round(float(value), digits)
    return value


def _period_metrics(entry):
    """선택 단계의 기간별 성과에서 화면에 쓰는 값만 남긴다."""
    out = {}
    counts = {"trades", "traded_symbols", "requested_symbols"}
    for name in ("development", "validation", "diagnostic"):
        block = (entry or {}).get(name) or {}
        out[name] = {field: (_int_if_integral(block.get(field)) if field in counts
                             else _round(block.get(field))) for field in PERIOD_FIELDS}
        out[name]["earnings_gate_status_counts"] = block.get("earnings_gate_status_counts")
        out[name]["max_drawdown_portfolio_5_equal_pct"] = _round(
            block.get("max_drawdown_portfolio_5_equal_pct"))
    return out


def _int_if_integral(value):
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def _stage_map(audit):
    """label(A, B, B+D …) -> 감사 단계 이름."""
    for cohort in ("KOSPI", "KOSDAQ", "TIGER_ETF", "SOXL"):
        ladder = ((audit.get("policy_ladder") or {}).get(cohort) or {}).get("stage_ladder") or {}
        mapping = {label: stage for label, stage in ladder.items()
                   if label in STAGE_ORDER and isinstance(stage, str)}
        if mapping:
            return mapping
    # policy_ladder가 없어도 화면은 계속 읽혀야 한다: 공식 라벨을 그대로 쓴다.
    return {"A": "legacy_only", "B": "legacy_risk_off_hard_gate",
            "B+D": "legacy_risk_off_disclosure", "C": "legacy_risk_off_trend_filter",
            "D": "legacy_risk_off_trend_disclosure",
            "B+D+E": "legacy_risk_off_disclosure_earnings"}


def _signal_row(row, selected_label):
    stages = row.get("stages") or {}
    order = [label for label in STAGE_ORDER if label in stages]
    selected = stages.get(selected_label) or {}
    # 선택 단계보다 뒤에 있는 단계가 막힌 것은 후보 자격과 무관하므로 차단 사유로 쓰지 않는다.
    upstream = order[:order.index(selected_label) + 1] if selected_label in order else order
    blocked = [(label, (stages.get(label) or {}).get("reason")) for label in upstream
               if (stages.get(label) or {}).get("accepted") is False
               and (stages.get(label) or {}).get("reason")]
    return {
        "ticker": row.get("ticker"),
        "signal_date": row.get("signal_date"),
        "market_regime": row.get("market_regime"),
        "disclosure_bad_60d": _int_if_integral(row.get("disclosure_bad_60d")),
        "disclosure_periodic_days_since": _int_if_integral(row.get("disclosure_periodic_days_since")),
        "stages": {label: {"accepted": (stages.get(label) or {}).get("accepted"),
                           "reason": (stages.get(label) or {}).get("reason")}
                   for label in order},
        "selected_stage": selected_label,
        "selected_accepted": selected.get("accepted"),
        "selected_reason": selected.get("reason"),
        "first_block": ({"label": blocked[0][0], "reason": blocked[0][1],
                         "text": REASON_LABELS.get(blocked[0][1], blocked[0][1])}
                        if blocked and selected.get("accepted") is not True else None),
    }


def _disclosure_coverage(audit):
    out = {}
    for cohort, block in (audit.get("disclosure_coverage") or {}).items():
        periods = (block or {}).get("periods") or {}
        summary = {}
        for name, features in periods.items():
            values = [v.get("non_missing_pct") for v in (features or {}).values()
                      if isinstance(v, dict) and v.get("non_missing_pct") is not None]
            summary[name] = {"features_n": len(values),
                             "min_non_missing_pct": _round(min(values)) if values else None}
        out[cohort] = summary
    return out


def _earnings_coverage(audit):
    source = audit.get("earnings_coverage") or {}
    if not source:
        return {}
    out = {"key_present": source.get("key_present"),
           "source": source.get("earnings_source"),
           "used_fields": source.get("used_fields"),
           "collection": {k: v for k, v in (source.get("collection") or {}).items()
                          if k in ("requested_symbols_n", "available_symbols_n", "attempted_symbols_n",
                                   "total_requests_n", "request_limit", "budget_exhausted",
                                   "cap_reached", "workers")},
           "cohorts": {}}
    for cohort in COHORT_LABELS:
        block = source.get(cohort) or {}
        if not block:
            continue
        periods = block.get("periods") or {}
        dev = periods.get("development") or {}
        out["cohorts"][cohort] = {
            "requested_symbols_n": block.get("requested_symbols_n"),
            "available_symbols_n": block.get("available_symbols_n"),
            "coverage_pct": _round(block.get("coverage_pct")),
            "missing_tickers": block.get("missing_tickers"),
            "not_applicable_tickers": block.get("not_applicable_tickers"),
            "development_non_missing_pct": {
                field: _round((dev.get(field) or {}).get("non_missing_pct"))
                for field in ("opm_pct", "op_yoy_pct", "earnings_momentum")},
        }
    return out


def _slb_coverage(audit):
    source = audit.get("slb_coverage") or {}
    if not source:
        return {}
    return {"source": source.get("source"),
            "backtest_status": source.get("backtest_status"),
            "reason": source.get("reason"),
            "available_range": source.get("available_range"),
            "traffic_limit": source.get("traffic_limit")}


def _earnings_ablation(audit):
    """E(실적)를 더했을 때 3방향 모델이 실제로 좋아졌는지만 남긴다."""
    out = {}
    for cohort, block in (audit.get("earnings_ablation") or {}).items():
        periods = (block or {}).get("periods") or {}
        deltas = {name: (periods.get(name) or {}).get("delta_plus_earnings_minus_plus_flow")
                  for name in ("development", "validation", "diagnostic")}
        if not any(deltas.values()):
            continue
        out[cohort] = {"status": block.get("status"),
                       "target_variant": block.get("target_variant"),
                       "delta_plus_earnings_minus_plus_flow": {
                           name: {k: _round(v) for k, v in (row or {}).items()}
                           for name, row in deltas.items()}}
    return out


def audit_attempt_summary(audit, *, run_id=None, head=None):
    """다른 실행의 감사가 B+D+E를 실제로 평가했는지 확인하는 요약.

    교체 판단에만 쓴다. 새 실행이 공시·실적 데이터를 못 받아 B+D+E를 미평가로
    남겼다면 그 실행으로 화면 데이터를 갈아치우지 않기 위해서다.
    """
    if not isinstance(audit, dict):
        return None
    cohorts = {}
    for cohort in COHORT_LABELS:
        ladder = (audit.get("policy_ladder") or {}).get(cohort) or {}
        step = ladder.get("legacy_risk_off_disclosure_earnings") or {}
        periods = step.get("periods") or {}
        disclosure = (((audit.get("disclosure_coverage") or {}).get(cohort) or {})
                      .get("periods") or {}).get("development") or {}
        periods_pct = [v.get("non_missing_pct") for v in disclosure.values()
                       if isinstance(v, dict) and v.get("non_missing_pct") is not None]
        earnings = ((audit.get("earnings_coverage") or {}).get(cohort) or {})
        cohorts[cohort] = {
            "bde_stage_status": step.get("status"),
            "bde_development_trades": (periods.get("development") or {}).get("trades"),
            "selected_policy": ladder.get("selected_policy"),
            "disclosure_development_min_non_missing_pct": (round(min(periods_pct), 2)
                                                            if periods_pct else None),
            "earnings_available_symbols_n": earnings.get("available_symbols_n"),
            "earnings_requested_symbols_n": earnings.get("requested_symbols_n"),
        }
    collection = (audit.get("earnings_coverage") or {}).get("collection") or {}
    return {"report_created_utc": audit.get("created_utc"), "run_id": run_id, "head": head,
            "report_status": audit.get("status"), "cohorts": cohorts,
            "earnings_collection": {k: collection.get(k) for k in
                                    ("attempted_symbols_n", "available_symbols_n",
                                     "requested_symbols_n", "budget_exhausted")}}


def _direction3(audit):
    """3방향(상승·횡보·하락) 분류 정확도. 매수 승률과 다른 지표라 따로 둔다."""
    out = {}
    for market, block in (audit.get("direction3_three_way") or {}).items():
        periods = {}
        for name in ("development", "validation", "diagnostic"):
            row = (block or {}).get(name) or {}
            if not row:
                continue
            periods[name] = {
                "evaluation_rows_n": row.get("evaluation_rows_n"),
                "accuracy_pct": _round(row.get("accuracy_pct")),
                "majority_class_baseline_accuracy_pct": _round(
                    row.get("majority_class_baseline_accuracy_pct")),
                "accuracy_delta_vs_majority_pp": _round(row.get("accuracy_delta_vs_majority_pp")),
                "multiclass_brier": _round(row.get("multiclass_brier")),
                "prior_brier": _round(row.get("prior_brier")),
                "class_metrics": {label: {"precision_pct": _round(v.get("precision_pct")),
                                          "recall_pct": _round(v.get("recall_pct"))}
                                  for label, v in (row.get("class_metrics") or {}).items()},
            }
        if periods:
            out[market] = periods
    return out


def build_summary(audit, *, source=None, generated_utc=None, newer_attempt=None):
    """전체 감사 보고서에서 B+D+E 화면용 요약을 만든다.

    원본 값은 그대로 옮기고 없는 값은 None으로 둔다. 계산·추정·보정은 하지 않는다.
    """
    if not isinstance(audit, dict):
        raise ValueError("감사 보고서가 JSON 객체가 아닙니다")
    mapping = _stage_map(audit)
    label_for_stage = {stage: label for label, stage in mapping.items()}
    summary = {
        "schema_version": SCHEMA_VERSION,
        "kind": "bde_recent_filter_and_candidates",
        "generated_utc": generated_utc or _now_utc().isoformat(),
        "research_only": True,
        "live_approved": False,
        "evaluation_warning": audit.get("evaluation_warning"),
        "stage_order": list(STAGE_ORDER),
        "stage_labels": {label: mapping.get(label) for label in STAGE_ORDER},
        "source": {
            "report_created_utc": audit.get("created_utc"),
            "report_status": audit.get("status"),
            "sample_start": audit.get("sample_start"),
            "windows": audit.get("windows"),
            "feature_timing": audit.get("feature_timing"),
            **(source or {}),
        },
        "cohorts": {},
    }
    if newer_attempt:
        summary["source"]["newer_attempt"] = newer_attempt
    for cohort in COHORT_LABELS:
        block = (audit.get("policy_ladder") or {}).get(cohort) or {}
        cohort_detail = (audit.get("cohorts") or {}).get(cohort) or {}
        selected_stage = block.get("selected_policy")
        selected_label = label_for_stage.get(selected_stage) or selected_stage
        selected_entry = block.get(selected_stage) or {} if selected_stage else {}
        signal_block = (audit.get("recent_signal_gate") or {}).get(cohort) or {}
        if not block and not signal_block:
            continue
        summary["cohorts"][cohort] = {
            "selected_policy": {
                "stage": selected_stage,
                "label": selected_label,
                "status": selected_entry.get("status"),
                "params": selected_entry.get("params"),
                "earnings_thresholds": selected_entry.get("earnings_thresholds"),
                "stage_definition": selected_entry.get("stage_definition"),
                "periods": _period_metrics(selected_entry.get("periods")),
                "earnings_coverage_pct": _round(selected_entry.get("earnings_coverage_pct")),
                "missing_feature_action": selected_entry.get("missing_feature_action"),
            },
            "recent_signal": _recent_signal(signal_block, mapping),
            "signals": [_signal_row(row, selected_label)
                        for row in (signal_block.get("latest_signals") or [])[:10]],
            "legacy_variant": (cohort_detail.get("legacy_selected_development_strategy")),
            "stage_params": {label: {"stage": mapping.get(label),
                                     "status": (block.get(mapping.get(label)) or {}).get("status"),
                                     "params": (block.get(mapping.get(label)) or {}).get("params"),
                                     "earnings_thresholds": (block.get(mapping.get(label)) or {}).get(
                                         "earnings_thresholds")}
                             for label in STAGE_ORDER if mapping.get(label) in block},
            "instruments": [{"ticker": row.get("ticker"), "name": row.get("name")}
                            for row in (cohort_detail.get("instruments") or [])],
        }
    detail = (audit.get("focused_universe_ablation") or {}).get("selection_rule") or {}
    summary["coverage"] = {
        "disclosure": _disclosure_coverage(audit),
        "earnings": _earnings_coverage(audit),
        "slb": _slb_coverage(audit),
    }
    summary["direction3"] = _direction3(audit)
    summary["earnings_ablation"] = _earnings_ablation(audit)
    summary["universe_note"] = {
        "policy_evaluated": (audit.get("focused_universe_ablation") or {}).get("policy_evaluated"),
        "benchmark_unchanged": (audit.get("focused_universe_ablation") or {}).get("benchmark_unchanged"),
        "selection_rule": detail or None,
    }
    summary["notes"] = _notes(summary)
    return summary


def _recent_signal(signal_block, mapping):
    accepted = signal_block.get("accepted_n_by_stage") or {}
    gate = signal_block.get("gate_passed_n_by_stage") or {}
    return {
        "status": signal_block.get("status"),
        "window_days": _int_if_integral(signal_block.get("window_days")),
        "window_start": signal_block.get("window_start"),
        "latest_signal_date": signal_block.get("latest_signal_date"),
        "signals_n": _int_if_integral(signal_block.get("signals_n")),
        "accepted_n_by_stage": {label: accepted.get(label) for label in STAGE_ORDER
                                if label in accepted},
        "gate_passed_n_by_stage": {label: gate.get(label) for label in STAGE_ORDER
                                   if label in gate},
        "stage_status": {label: (signal_block.get("stage_status") or {}).get(stage)
                         for label, stage in mapping.items()
                         if stage in (signal_block.get("stage_status") or {})},
        "dropped_by_disclosure": (signal_block.get("b_plus_d_filter") or {}).get("dropped") or [],
    }


def _notes(summary):
    """화면에 반드시 함께 보여야 하는 한계. 데이터에서 직접 확인된 것만 적는다."""
    notes = []
    attempt = (summary.get("source") or {}).get("newer_attempt") or {}
    if attempt:
        attempt_cohorts = attempt.get("cohorts") or {}
        failed = [COHORT_LABELS.get(c, c) for c, v in attempt_cohorts.items()
                  if (v or {}).get("bde_stage_status") not in (None, "EVALUATED")]
        if failed:
            coverage_values = [(v or {}).get("disclosure_development_min_non_missing_pct")
                               for v in attempt_cohorts.values()]
            no_disclosure = bool(coverage_values) and all(
                c in (0, 0.0, None) for c in coverage_values)
            reason = ("공시 데이터가 확보되지 않아(개발구간 결측률 0%)" if no_disclosure
                      else "상류 단계 데이터가 없어")
            kept = str((summary.get("source") or {}).get("report_created_utc"))[:10]
            notes.append(
                f"더 새로운 감사 시도({str(attempt.get('report_created_utc'))[:10]}, "
                f"run {attempt.get('run_id')})는 {reason} {'·'.join(failed)}의 B+D+E 단계를 "
                f"평가하지 못했습니다. 그래서 화면은 마지막으로 실제 평가된 {kept} 결과를 유지합니다.")
    coverage = (summary.get("coverage") or {}).get("earnings") or {}
    collection = coverage.get("collection") or {}
    if collection.get("budget_exhausted"):
        notes.append(
            f"실적 수집 요청예산이 소진되어 {collection.get('available_symbols_n')}"
            f"/{collection.get('requested_symbols_n')}종목만 확보했습니다. "
            "실적이 없는 종목은 E단계가 판정 보류(ABSTAIN)로 처리됩니다.")
    slb = (summary.get("coverage") or {}).get("slb") or {}
    if slb and slb.get("backtest_status") and slb.get("backtest_status") != "EVALUATED":
        notes.append("SLB(공매도 잔고)는 백테스트에 반영되지 않았습니다: "
                     + str(slb.get("backtest_status")))
    earnings = (summary.get("coverage") or {}).get("earnings") or {}
    if earnings and earnings.get("key_present") is False:
        notes.append("OpenDART 키가 없어 실적(E) 단계는 판정 보류(ABSTAIN)로만 기록됩니다.")
    for cohort, block in (summary.get("earnings_ablation") or {}).items():
        dev = ((block.get("delta_plus_earnings_minus_plus_flow") or {}).get("development") or {})
        brier = dev.get("brier")
        auroc = dev.get("auroc")
        if brier is not None and auroc is not None and abs(brier) < 0.001 and abs(auroc) < 0.01:
            notes.append(f"{COHORT_LABELS.get(cohort, cohort)}: 실적 피처를 더해도 3방향 모델 성능 변화가"
                         " 사실상 없습니다(Brier·AUROC 차이 < 0.001/0.01).")
    for cohort, block in (summary.get("cohorts") or {}).items():
        val = ((block.get("selected_policy") or {}).get("periods") or {}).get("validation") or {}
        if val.get("trades") is not None and val.get("trades") < 10:
            notes.append(f"{COHORT_LABELS.get(cohort, cohort)}: 선택 정책 검증 거래가 {val['trades']}건뿐이라"
                         " 검증 성과를 확정으로 보지 않습니다.")
    notes.append("모든 수치는 이미 열람한 과거 구간의 연구 적합도이며 미래 승률 보장이 아닙니다.")
    return notes


def write_summary(summary, path=DEFAULT_SUMMARY_PATH):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    return path


def load_summary(path=DEFAULT_SUMMARY_PATH):
    path = Path(path)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def load_live(path="data/live-candidates-latest.json"):
    return load_summary(path) if Path(path).exists() else None


def summary_age_days(summary, now=None):
    stamp = (summary or {}).get("source", {}).get("report_created_utc")
    if not stamp:
        return None
    try:
        created = datetime.fromisoformat(str(stamp).replace("Z", "+00:00"))
    except ValueError:
        return None
    if created.tzinfo is None:
        created = created.replace(tzinfo=timezone.utc)
    return ((now or _now_utc()) - created).total_seconds() / 86400.0


def live_headline(live):
    """오늘 화면의 결론: 단계별 통과 종목 수."""
    if not live:
        return None
    rows = []
    for cohort, block in (live.get("cohorts") or {}).items():
        counts = block.get("counts_by_stage") or {}
        rows.append({"cohort": cohort, "counts": counts,
                     "signal_bar_date": block.get("signal_bar_date"),
                     "raw_signal_n": block.get("raw_signal_n")})
    return {"generated_at_kst": live.get("generated_at_kst"),
            "signal_bar_date": live.get("signal_bar_date"),
            "cohorts": rows}


def render(st, summary, live=None):
    """앱 화면을 그린다. streamlit은 호출 시점에만 가져온다."""
    import pandas as pd

    st.subheader("🧭 B+D+E 실전 필터 · 최근 후보")
    st.caption("B(DART 공시) + RISK_OFF 하드게이트 + E(OpenDART 실적). 연구용이며 실전 매수 승인이 아닙니다.")
    if not summary:
        st.info("B+D+E 요약 파일이 없습니다. `python bde_dashboard.py <audit-63-latest.json>`로 생성하세요.")
        return
    age = summary_age_days(summary)
    source = summary.get("source") or {}
    attempt = source.get("newer_attempt") or {}
    head_left, head_mid, head_right = st.columns(3)
    head_left.metric("요약 생성", str(summary.get("generated_utc", "—"))[:16].replace("T", " ") + " UTC")
    head_mid.metric("감사 보고서 기준", str(source.get("report_created_utc", "—"))[:16].replace("T", " ")
                    + (f" (경과 {age:.1f}일)" if age is not None else ""))
    if live:
        head_right.metric("오늘 신호 기준일", str(live.get("signal_bar_date") or "—"))
    else:
        head_right.metric("오늘 신호 기준일", "없음")
    if age is not None and age > STALE_AFTER_DAYS:
        st.warning(f"B+D+E 요약이 {age:.0f}일 지났습니다. 최신 감사 아티팩트로 다시 생성하기 전까지는 "
                   "이전 판정입니다.")
    if attempt:
        statuses = {COHORT_LABELS.get(c, c): (v or {}).get("bde_stage_status")
                    for c, v in (attempt.get("cohorts") or {}).items()}
        st.warning("더 새로운 감사 시도(" + str(attempt.get("report_created_utc"))[:16].replace("T", " ")
                   + f', run {attempt.get("run_id")})는 B+D+E를 평가하지 못했습니다: '
                   + ", ".join(f"{k} {v}" for k, v in statuses.items())
                   + ". 그래서 화면은 마지막으로 실제 평가된 결과를 보여줍니다.")

    if live:
        st.markdown("**오늘의 후보 단계별 통과 수**")
        live_rows = []
        for cohort, block in (live.get("cohorts") or {}).items():
            counts = block.get("counts_by_stage") or {}
            live_rows.append({"시장": COHORT_LABELS.get(cohort, cohort),
                              "기준일": block.get("signal_bar_date"),
                              "원신호": block.get("raw_signal_n"),
                              "시장국면": (block.get("diagnostics") or {}).get("market_regime"),
                              **{label: counts.get(label) for label in STAGE_ORDER}})
        if live_rows:
            st.dataframe(pd.DataFrame(live_rows), hide_index=True, use_container_width=True)
            st.caption(f'생성 {str(live.get("generated_at_kst", "—"))[:19]} · 기준일 '
                       f'{live.get("signal_bar_date", "—")} · 원신호는 선택 변형 조건을 통과한 건수입니다.')
            bar = live.get("signal_bar_date")
            if bar:
                try:
                    age_days = (pd.Timestamp.now(tz="Asia/Seoul").normalize()
                                - pd.Timestamp(bar).tz_localize("Asia/Seoul")).days
                    if age_days > 3:
                        st.warning(f"오늘 후보가 {age_days}일 전 종가 기준입니다. "
                                   "`python live_screen.py`로 다시 실행해야 최신입니다.")
                except Exception:
                    pass
        diagnostics = []
        for cohort, block in (live.get("cohorts") or {}).items():
            diag = block.get("diagnostics") or {}
            if not diag:
                continue
            diagnostics.append({"시장": COHORT_LABELS.get(cohort, cohort),
                                "검사 종목": diag.get("symbols_checked"),
                                "시장국면": diag.get("market_regime"),
                                "지수 MA200 갭 %": diag.get("market_index_ma200_gap_pct"),
                                "변동성비(20/60)": diag.get("market_volatility_ratio"),
                                "VIX 확장백분위": diag.get("vix_expanding_percentile"),
                                "코호트 프록시 갭 %": diag.get("cohort_proxy_ma200_gap_pct")})
        if diagnostics:
            st.dataframe(pd.DataFrame(diagnostics), hide_index=True, use_container_width=True)
        misses = {c: [m for m in ((b.get("diagnostics") or {}).get("near_misses") or [])]
                  for c, b in (live.get("cohorts") or {}).items()}
        if any(misses.values()):
            st.markdown("**진입 패턴만 미충족한 근접 종목**")
            rows = [{"시장": COHORT_LABELS.get(c, c), "종목": m.get("ticker"),
                     "종가": m.get("close"), "20일 고점 대비 %": m.get("breakout_pct"),
                     "RSI": m.get("rsi"), "ADX": m.get("adx"), "거래량비": m.get("volume_ratio")}
                    for c, items in misses.items() for m in items]
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        conditions = []
        for cohort, block in (live.get("cohorts") or {}).items():
            diag = block.get("diagnostics") or {}
            if diag.get("condition_failures"):
                conditions.append({"시장": COHORT_LABELS.get(cohort, cohort),
                                   **diag["condition_failures"]})
        if conditions:
            st.caption("전역 진입조건 실패 종목 수(검사 대상 대비). 원신호 0건의 원인을 설명합니다.")
            st.dataframe(pd.DataFrame(conditions), hide_index=True, use_container_width=True)
        for cohort, block in (live.get("cohorts") or {}).items():
            rows = block.get("today_signals") or []
            if not rows:
                continue
            table = []
            for row in rows:
                selected = (row.get("stages") or {}).get(block.get("selected_label") or "") or {}
                table.append({"시장": COHORT_LABELS.get(cohort, cohort),
                              "종목": f'{row.get("name") or ""} ({row.get("ticker")})',
                              "전략": row.get("strategy"), "종가": row.get("close"),
                              "신호일": row.get("signal_date"),
                              "선택 단계": block.get("selected_label"),
                              "판정": ("통과" if selected.get("accepted") else
                                       REASON_LABELS.get(selected.get("reason"), selected.get("reason"))),
                              "시장국면": row.get("market_regime"),
                              "실적데이터": row.get("earnings_status")})
            st.dataframe(pd.DataFrame(table), hide_index=True, use_container_width=True)
        if live.get("notes"):
            for note in live["notes"]:
                st.caption("· " + note)

    st.markdown("**감사 기준 최근 신호 판정 (단계별 통과·차단)**")
    funnel = []
    for cohort, block in (summary.get("cohorts") or {}).items():
        recent = block.get("recent_signal") or {}
        accepted = recent.get("accepted_n_by_stage") or {}
        gate = recent.get("gate_passed_n_by_stage") or {}
        funnel.append({"시장": COHORT_LABELS.get(cohort, cohort),
                       "선택 정책": (block.get("selected_policy") or {}).get("label"),
                       "최근 신호일": recent.get("latest_signal_date"),
                       "창(일)": recent.get("window_days"),
                       "신호 수": recent.get("signals_n"),
                       "게이트 통과(A/B/B+D/C/D/B+D+E)": " / ".join(
                           f"{label} {gate.get(label) if gate.get(label) is not None else '—'}"
                           for label in STAGE_ORDER),
                       "채택 수(A/B/B+D/C/D/B+D+E)": " / ".join(
                           f"{label} {accepted.get(label) if accepted.get(label) is not None else '—'}"
                           for label in STAGE_ORDER)})
    if funnel:
        st.dataframe(pd.DataFrame(funnel), hide_index=True, use_container_width=True)

    signals = []
    for cohort, block in (summary.get("cohorts") or {}).items():
        label = (block.get("selected_policy") or {}).get("label")
        for row in block.get("signals") or []:
            first = row.get("first_block") or {}
            signals.append({"시장": COHORT_LABELS.get(cohort, cohort), "종목": row.get("ticker"),
                            "신호일": row.get("signal_date"), "시장국면": row.get("market_regime"),
                            "악성공시 60일": row.get("disclosure_bad_60d"),
                            "정기보고서 경과(거래일)": row.get("disclosure_periodic_days_since"),
                            "선택 정책": label,
                            "선택 단계 판정": ("통과" if row.get("selected_accepted")
                                          else REASON_LABELS.get(row.get("selected_reason"),
                                                                 row.get("selected_reason"))),
                            "차단 단계": (f'{first.get("label")} · {first.get("text")}'
                                       if first else "—")})
    if signals:
        st.markdown("**최근 신호별 판정 (최근 10건)**")
        st.dataframe(pd.DataFrame(signals), hide_index=True, use_container_width=True)
        st.caption("차단 사유 코드는 감사 게이트가 돌려준 값 그대로이며, 게이트 기준을 낮추지 않았습니다.")

    st.markdown("**선택 정책 백테스트 성과 (개발·검증·진단)**")
    periods = []
    for cohort, block in (summary.get("cohorts") or {}).items():
        policy = block.get("selected_policy") or {}
        for name, stats in (policy.get("periods") or {}).items():
            periods.append({"시장": COHORT_LABELS.get(cohort, cohort),
                            "정책": policy.get("label"), "기간": PERIOD_LABELS.get(name, name),
                            "거래": stats.get("trades"), "승률 %": stats.get("win_rate_pct"),
                            "거래당 순손익 %": stats.get("mean_net_pct"),
                            "손익비": stats.get("profit_factor"),
                            "연속 최대낙폭 %": stats.get("max_drawdown_sequence_pct"),
                            "신호비중 %": stats.get("coverage_pct")})
    if periods:
        st.dataframe(pd.DataFrame(periods), hide_index=True, use_container_width=True)
        st.caption("이는 매수 정책의 과거 성과이며, 아래 3방향 분류 정확도와 다른 지표입니다.")

    direction = summary.get("direction3") or {}
    if direction:
        st.markdown("**3방향(상승·횡보·하락) 분류 정확도 — 매수 승률과 별개**")
        rows = []
        for market, block in direction.items():
            for name, stats in block.items():
                rows.append({"시장": DIRECTION_MARKETS.get(market, market),
                             "기간": PERIOD_LABELS.get(name, name),
                             "표본": stats.get("evaluation_rows_n"),
                             "정확도 %": stats.get("accuracy_pct"),
                             "다수클래스 기준 %": stats.get("majority_class_baseline_accuracy_pct"),
                             "기준 대비 %p": stats.get("accuracy_delta_vs_majority_pp"),
                             "Brier": stats.get("multiclass_brier"),
                             "기준 Brier": stats.get("prior_brier")})
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        st.caption("정확도가 다수클래스 기준보다 높지 않으면 방향 예측 우위가 없습니다. Brier는 낮을수록 좋습니다.")

    coverage = summary.get("coverage") or {}
    st.markdown("**데이터 확보 현황**")
    earnings = coverage.get("earnings") or {}
    disclosure = coverage.get("disclosure") or {}
    slb = coverage.get("slb") or {}
    cover_rows = []
    for cohort in COHORT_LABELS:
        earn = ((earnings.get("cohorts") or {}).get(cohort)) or {}
        disc = disclosure.get(cohort) or {}
        cover_rows.append({"시장": COHORT_LABELS[cohort],
                           "공시(DART) 개발구간 최소 결측률 %": (
                               (disc.get("development") or {}).get("min_non_missing_pct")
                               if cohort in ("KOSPI", "KOSDAQ") else "—"),
                           "실적 종목 확보": (f'{earn.get("available_symbols_n")}/{earn.get("requested_symbols_n")}'
                                          if earn and cohort in ("KOSPI", "KOSDAQ") else "—"),
                           "실적 종목 확보율 %": (earn.get("coverage_pct")
                                                if cohort in ("KOSPI", "KOSDAQ") else "—"),
                           "OPM 결측률 %": ((earn.get("development_non_missing_pct") or {}).get("opm_pct")
                                          if cohort in ("KOSPI", "KOSDAQ") else "—")})
    if cover_rows:
        st.dataframe(pd.DataFrame(cover_rows), hide_index=True, use_container_width=True)
        st.caption("ETF·해외종목(TIGER ETF·SOXL)은 공시·실적 필터 대상이 아닙니다.")
    if slb:
        st.caption(f'SLB(공매도 잔고): {slb.get("backtest_status")} — {slb.get("reason")}')
    if earnings:
        st.caption("실적(E) 소스: " + str(earnings.get("source")) +
                   f' · OpenDART 키 {("있음" if earnings.get("key_present") else "없음")}')

    for note in summary.get("notes") or []:
        st.caption("· " + note)
    st.caption("원본: " + str((summary.get("source") or {}).get("audit_report", "data/audit-63-latest.json")))


def _report_label(path):
    """저장소 경로는 그대로, 저장소 밖(CI 아티팩트) 경로는 파일명만 남긴다."""
    path = Path(path)
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except Exception:
        return "audit-63 artifact: " + path.name


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="B+D+E 감사 요약(data/bde-latest.json) 생성")
    parser.add_argument("audit", nargs="?", default=str(DEFAULT_AUDIT_PATH),
                        help="감사 보고서 경로 (기본 data/audit-63-latest.json)")
    parser.add_argument("--out", default=str(DEFAULT_SUMMARY_PATH))
    parser.add_argument("--run-id", default=None, help="감사를 만든 GitHub Actions run id")
    parser.add_argument("--head", default=None, help="감사를 만든 커밋 SHA")
    parser.add_argument("--branch", default=None)
    parser.add_argument("--attempt-audit", default=None,
                        help="더 새로운 감사 시도 보고서 경로(교체 대신 비교용으로 기록)")
    parser.add_argument("--attempt-run-id", default=None)
    parser.add_argument("--status", action="store_true", help="커밋된 요약의 기준 시각만 확인")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.status:
        summary = load_summary(args.out)
        if not summary:
            print("NO_SUMMARY " + str(args.out))
            return 1
        age = summary_age_days(summary)
        print(json.dumps({"path": args.out,
                          "generated_utc": summary.get("generated_utc"),
                          "report_created_utc": (summary.get("source") or {}).get("report_created_utc"),
                          "age_days": round(age, 2) if age is not None else None,
                          "stale": bool(age is not None and age > STALE_AFTER_DAYS),
                          "cohorts": {c: (b.get("selected_policy") or {}).get("label")
                                      for c, b in (summary.get("cohorts") or {}).items()},
                          "notes": summary.get("notes")}, ensure_ascii=False, indent=1))
        return 0
    path = Path(args.audit)
    if not path.exists():
        print(f"감사 보고서가 없습니다: {path}")
        return 2
    audit = json.loads(path.read_text(encoding="utf-8"))
    source = {"audit_report": _report_label(path),
              "run_id": args.run_id, "head": args.head, "branch": args.branch}
    attempt = None
    if args.attempt_audit:
        attempt_path = Path(args.attempt_audit)
        if attempt_path.exists():
            attempt = audit_attempt_summary(json.loads(attempt_path.read_text(encoding="utf-8")),
                                            run_id=args.attempt_run_id)
        else:
            print("ATTEMPT_REPORT_MISSING " + str(attempt_path))
    summary = build_summary(audit, source=source, newer_attempt=attempt)
    out = write_summary(summary, args.out)
    print("WROTE " + str(out) + " " + json.dumps({
        "bytes": out.stat().st_size,
        "report_created_utc": (summary.get("source") or {}).get("report_created_utc"),
        "selected_policy": {c: (b.get("selected_policy") or {}).get("label")
                            for c, b in summary["cohorts"].items()}}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
