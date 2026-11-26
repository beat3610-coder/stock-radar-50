import json
from pathlib import Path

import pytest

import bde_dashboard as bd


def _audit():
    """감사 보고서의 실제 구조를 줄인 표본. 값을 지어내지 않고 구조만 재현한다."""
    return {
        "created_utc": "2026-11-25T14:35:29+00:00",
        "status": "RESEARCH_ONLY_NOT_LIVE_APPROVED",
        "sample_start": "2016-01-01",
        "windows": {"development_end": "2025-11-01", "validation_end": "2026-04-01",
                    "diagnostic_end": "2026-11-01"},
        "policy_ladder": {
            "KOSPI": {
                "selected_policy": "legacy_risk_off_disclosure_earnings",
                "stage_ladder": {"A": "legacy_only", "B": "legacy_risk_off_hard_gate",
                                 "B+D": "legacy_risk_off_disclosure",
                                 "C": "legacy_risk_off_trend_filter",
                                 "D": "legacy_risk_off_trend_disclosure",
                                 "B+D+E": "legacy_risk_off_disclosure_earnings"},
                "legacy_risk_off_disclosure_earnings": {
                    "status": "EVALUATED",
                    "params": {"disclosure_bad_60d_max": 0, "disclosure_periodic_days_since_floor": 20,
                               "earnings_momentum_min": None, "op_yoy_min": None, "opm_min": 0},
                    "earnings_thresholds": {"opm_min": 0},
                    "earnings_coverage_pct": 100.0,
                    "missing_feature_action": "ABSTAIN; unavailable earnings do not reject a legacy entry.",
                    "periods": {
                        "development": {"trades": 31.0, "win_rate_pct": 51.61, "mean_net_pct": 1.456,
                                        "profit_factor": 1.737, "max_drawdown_sequence_pct": 19.6,
                                        "coverage_pct": 70.0, "traded_symbols": 21.0,
                                        "requested_symbols": 30.0, "small_sample": False,
                                        "earnings_gate_status_counts": {"REJECT": 2, "PASS": 19,
                                                                        "ABSTAIN": 12}},
                        "validation": {"trades": 6.0, "win_rate_pct": 50.0, "mean_net_pct": 3.843,
                                       "small_sample": True},
                        "diagnostic": {"trades": 0.0},
                    },
                },
            },
            "KOSDAQ": {"selected_policy": None},
        },
        "recent_signal_gate": {
            "KOSPI": {
                "status": "EVALUATED", "window_days": 14, "window_start": "2026-08-20",
                "latest_signal_date": "2026-09-03", "signals_n": 1,
                "accepted_n_by_stage": {"A": 1, "B": 1, "B+D": 0, "C": 1, "D": 1, "B+D+E": 0},
                "gate_passed_n_by_stage": {"A": 1, "B": 1, "B+D": 0, "C": 1, "D": 1, "B+D+E": 0},
                "stage_status": {"legacy_only": "EVALUATED"},
                "b_plus_d_filter": {"dropped": [{"ticker": "055550", "signal_date": "2026-09-03",
                                                 "reason": "PERIODIC_REPORT_TOO_RECENT",
                                                 "disclosure_bad_60d": 0.0,
                                                 "disclosure_periodic_days_since": 13}]},
                "latest_signals": [{
                    "ticker": "055550", "signal_date": "2026-09-03", "market_regime": "RISK_ON",
                    "disclosure_bad_60d": 0.0, "disclosure_periodic_days_since": 13.0,
                    "stages": {"A": {"accepted": True, "reason": None},
                               "B": {"accepted": True, "reason": None},
                               "B+D": {"accepted": False, "reason": "PERIODIC_REPORT_TOO_RECENT"},
                               "C": {"accepted": True, "reason": None},
                               "D": {"accepted": True, "reason": None},
                               "B+D+E": {"accepted": False, "reason": "PERIODIC_REPORT_TOO_RECENT"}}}],
            },
            "TIGER_ETF": {"status": "NO_SIGNALS"},
        },
        "disclosure_coverage": {"KOSPI": {"periods": {"development": {
            "disclosure_bad_60d": {"non_missing_pct": 100.0},
            "disclosure_periodic_days_since": {"non_missing_pct": 99.5}}}}},
        "earnings_coverage": {
            "key_present": True,
            "earnings_source": "OpenDART fnlttSinglAcntAll.json",
            "used_fields": ["opm_pct"],
            "collection": {"requested_symbols_n": 60, "available_symbols_n": 60,
                           "total_requests_n": 0, "request_limit": 6000, "workers": 4},
            "KOSPI": {"requested_symbols_n": 30, "available_symbols_n": 30, "coverage_pct": 100.0,
                      "missing_tickers": [],
                      "periods": {"development": {"opm_pct": {"non_missing_pct": 67.2}}}},
        },
        "slb_coverage": {"source": "data.go.kr B552481/SlbSvc XML",
                         "backtest_status": "NOT_EVALUATED",
                         "reason": "The dealing endpoint exposes only the latest five trading days",
                         "available_range": {"dealing_last_date": "20261124"}},
        "earnings_ablation": {"KOSPI": {"status": "EVALUATED_SAME_MODEL_TARGET_AND_SHARED_ROWS",
                                        "target_variant": "pooled",
                                        "periods": {"development": {
                                            "delta_plus_earnings_minus_plus_flow": {
                                                "auroc": 0.000506, "brier": -0.000003}}}}},
        "direction3_three_way": {"KOSPI": {
            "development": {"evaluation_rows_n": 10510, "accuracy_pct": 35.03,
                            "majority_class_baseline_accuracy_pct": 35.78,
                            "accuracy_delta_vs_majority_pp": -0.75,
                            "multiclass_brier": 0.675, "prior_brier": 0.664,
                            "class_metrics": {"UP": {"precision_pct": 34.5, "recall_pct": 55.9}}},
            "validation": {"evaluation_rows_n": 465, "accuracy_pct": 49.89,
                           "majority_class_baseline_accuracy_pct": 49.68,
                           "accuracy_delta_vs_majority_pp": 0.22,
                           "multiclass_brier": 0.645, "prior_brier": 0.624},
        }},
    }


def test_build_summary_maps_stage_labels_and_selected_policy():
    summary = bd.build_summary(_audit())
    assert summary["research_only"] is True
    assert summary["live_approved"] is False
    assert summary["stage_order"] == list(bd.STAGE_ORDER)
    assert summary["stage_labels"]["B+D+E"] == "legacy_risk_off_disclosure_earnings"
    policy = summary["cohorts"]["KOSPI"]["selected_policy"]
    assert policy["label"] == "B+D+E"
    assert policy["params"]["opm_min"] == 0
    assert policy["periods"]["development"]["trades"] == 31
    assert policy["periods"]["development"]["win_rate_pct"] == 51.61
    assert policy["periods"]["development"]["earnings_gate_status_counts"]["ABSTAIN"] == 12


def test_build_summary_keeps_recent_signal_funnel_and_reasons():
    summary = bd.build_summary(_audit())
    recent = summary["cohorts"]["KOSPI"]["recent_signal"]
    assert recent["latest_signal_date"] == "2026-09-03"
    assert recent["accepted_n_by_stage"]["B+D"] == 0
    assert recent["dropped_by_disclosure"][0]["reason"] == "PERIODIC_REPORT_TOO_RECENT"
    signal = summary["cohorts"]["KOSPI"]["signals"][0]
    assert signal["ticker"] == "055550"
    assert signal["selected_accepted"] is False
    assert signal["first_block"]["label"] == "B+D"
    assert signal["first_block"]["text"] == bd.REASON_LABELS["PERIODIC_REPORT_TOO_RECENT"]


def test_build_summary_reports_data_coverage_and_limits():
    summary = bd.build_summary(_audit())
    coverage = summary["coverage"]
    assert coverage["disclosure"]["KOSPI"]["development"]["min_non_missing_pct"] == 99.5
    assert coverage["earnings"]["cohorts"]["KOSPI"]["available_symbols_n"] == 30
    assert coverage["slb"]["backtest_status"] == "NOT_EVALUATED"
    notes = " ".join(summary["notes"])
    assert "SLB" in notes
    assert "검증 거래가 6건" in notes
    assert "미래 승률 보장이 아닙니다" in notes


def test_direction_accuracy_is_kept_separate_from_buy_win_rate():
    summary = bd.build_summary(_audit())
    direction = summary["direction3"]["KOSPI"]["development"]
    assert direction["accuracy_pct"] == 35.03
    assert "buy" not in json.dumps(summary["direction3"])
    buy = summary["cohorts"]["KOSPI"]["selected_policy"]["periods"]["development"]["win_rate_pct"]
    assert buy == 51.61 and buy != direction["accuracy_pct"]


def test_build_summary_survives_incomplete_reports():
    summary = bd.build_summary({"created_utc": "2026-11-25T00:00:00+00:00"})
    assert summary["cohorts"] == {}
    assert summary["direction3"] == {}
    assert isinstance(summary["notes"], list)
    assert summary["stage_labels"]["A"] == "legacy_only"


def test_build_summary_rejects_non_object_report():
    with pytest.raises(ValueError):
        bd.build_summary([1, 2, 3])


def test_summary_age_days_and_staleness(tmp_path):
    summary = bd.build_summary(_audit())
    age = bd.summary_age_days(summary)
    assert age is not None and age > 0
    old = {"source": {"report_created_utc": "2020-01-01T00:00:00+00:00"}}
    assert bd.summary_age_days(old) > bd.STALE_AFTER_DAYS
    assert bd.summary_age_days({}) is None


def test_write_and_load_roundtrip(tmp_path):
    summary = bd.build_summary(_audit())
    path = bd.write_summary(summary, tmp_path / "bde.json")
    loaded = bd.load_summary(path)
    assert loaded["cohorts"]["KOSPI"]["selected_policy"]["label"] == "B+D+E"
    assert bd.load_summary(tmp_path / "missing.json") is None
    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")
    assert bd.load_summary(broken) is None


def test_live_headline_reads_stage_counts():
    live = {"generated_at_kst": "2026-11-26 16:10:00+09:00", "signal_bar_date": "2026-11-26",
            "cohorts": {"KOSPI": {"counts_by_stage": {"A": 2, "B+D+E": 0}, "signal_bar_date": "2026-11-26",
                                  "raw_signal_n": 2}}}
    headline = bd.live_headline(live)
    assert headline["cohorts"][0]["counts"]["A"] == 2
    assert bd.live_headline(None) is None


def test_audit_attempt_summary_detects_unevaluated_bde_stage():
    attempt = bd.audit_attempt_summary({
        "created_utc": "2026-11-26T06:23:37+00:00",
        "status": "RESEARCH_ONLY_NOT_LIVE_APPROVED",
        "policy_ladder": {"KOSPI": {
            "selected_policy": "legacy_risk_off_trend_filter",
            "legacy_risk_off_disclosure_earnings": {
                "status": "UPSTREAM_DISCLOSURE_NOT_EVALUATED",
                "periods": {"development": {"trades": 0}}}}},
        "disclosure_coverage": {"KOSPI": {"periods": {"development": {
            "disclosure_bad_60d": {"non_missing_pct": 0.0}}}}},
        "earnings_coverage": {"KOSPI": {"available_symbols_n": 23, "requested_symbols_n": 30},
                              "collection": {"attempted_symbols_n": 23, "available_symbols_n": 23,
                                             "requested_symbols_n": 60, "budget_exhausted": True}},
    }, run_id="45767809305")
    assert attempt["cohorts"]["KOSPI"]["bde_stage_status"] == "UPSTREAM_DISCLOSURE_NOT_EVALUATED"
    assert attempt["cohorts"]["KOSPI"]["disclosure_development_min_non_missing_pct"] == 0.0
    assert attempt["earnings_collection"]["budget_exhausted"] is True
    assert bd.audit_attempt_summary([1, 2]) is None


def test_unevaluated_newer_attempt_keeps_previous_summary_and_says_why():
    summary = bd.build_summary(_audit(), newer_attempt={
        "report_created_utc": "2026-11-26T06:23:37+00:00", "run_id": "45767809305",
        "cohorts": {"KOSPI": {"bde_stage_status": "UPSTREAM_DISCLOSURE_NOT_EVALUATED",
                              "disclosure_development_min_non_missing_pct": 0.0}}})
    assert summary["cohorts"]["KOSPI"]["selected_policy"]["label"] == "B+D+E"
    note = summary["notes"][0]
    assert "45767809305" in note and "공시 데이터가 확보되지 않아" in note
    assert "2026-11-25" in note


def test_cli_writes_summary_and_status(tmp_path):
    audit_path = tmp_path / "audit-63-latest.json"
    audit_path.write_text(json.dumps(_audit()), encoding="utf-8")
    out = tmp_path / "bde-latest.json"
    code = bd.main([str(audit_path), "--out", str(out), "--run-id", "45635628647"])
    assert code == 0
    written = json.loads(out.read_text(encoding="utf-8"))
    assert written["source"]["run_id"] == "45635628647"
    assert written["source"]["audit_report"].endswith("audit-63-latest.json")
    assert bd.main(["--status", "--out", str(out)]) == 0
    assert bd.main(["--status", "--out", str(tmp_path / "none.json")]) == 1
    assert bd.main([str(tmp_path / "absent.json"), "--out", str(out)]) == 2
