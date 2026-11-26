import json

import pytest

import live_screen as ls


def _summary():
    return {
        "schema_version": 1,
        "generated_utc": "2026-11-26T07:41:35+00:00",
        "source": {"report_created_utc": "2026-11-25T14:35:29+00:00", "sample_start": "2016-01-01"},
        "stage_order": ["A", "B", "B+D", "C", "D", "B+D+E"],
        "stage_labels": {"A": "legacy_only", "B": "legacy_risk_off_hard_gate",
                         "B+D": "legacy_risk_off_disclosure", "C": "legacy_risk_off_trend_filter",
                         "D": "legacy_risk_off_trend_disclosure",
                         "B+D+E": "legacy_risk_off_disclosure_earnings"},
        "cohorts": {
            "KOSPI": {
                "selected_policy": {"label": "B+D+E"},
                "legacy_variant": "trend_breakout",
                "stage_params": {
                    "A": {"stage": "legacy_only", "params": {}},
                    "B": {"stage": "legacy_risk_off_hard_gate", "params": {"rule": "x"}},
                    "B+D": {"stage": "legacy_risk_off_disclosure",
                            "params": {"disclosure_bad_60d_max": 0,
                                       "disclosure_periodic_days_since_floor": 20}},
                    "B+D+E": {"stage": "legacy_risk_off_disclosure_earnings",
                              "params": {"disclosure_bad_60d_max": 0,
                                         "disclosure_periodic_days_since_floor": 20,
                                         "opm_min": 0}},
                },
                "instruments": [{"ticker": "005930", "name": "삼성전자"}],
            }
        },
    }


def _row(**overrides):
    row = {"market_regime": "RISK_ON", "disclosure_bad_60d": 0.0,
           "disclosure_periodic_days_since": 30.0, "market_ma200_gap_pct": -3.0,
           "signal_trend_200_pct": 5.0, "relative_strength_60d_pct": 2.0,
           "signal_volatility_20d_pct": 1.5, "earnings_data_available": 1,
           "earnings_momentum": 12.0, "op_yoy_pct": 8.0, "opm_pct": 6.0}
    row.update(overrides)
    return row


def test_load_config_reads_gate_parameters_from_summary(tmp_path):
    path = tmp_path / "bde-latest.json"
    path.write_text(json.dumps(_summary()), encoding="utf-8")
    config = ls.load_config(path)
    assert config["sample_start"] == "2016-01-01"
    assert config["stage_labels"]["B+D+E"] == "legacy_risk_off_disclosure_earnings"
    assert config["cohorts"]["KOSPI"]["legacy_variant"] == "trend_breakout"
    assert config["cohorts"]["KOSPI"]["stage_params"]["B+D"]["disclosure_periodic_days_since_floor"] == 20


def test_load_config_refuses_missing_summary(tmp_path):
    with pytest.raises(SystemExit):
        ls.load_config(tmp_path / "absent.json")


def test_stage_decisions_use_audit_gate_and_never_relax_it():
    summary = _summary()
    labels = summary["stage_labels"]
    params = {label: entry["params"] for label, entry in summary["cohorts"]["KOSPI"]["stage_params"].items()}
    decisions = ls._stage_decisions(_row(), summary["stage_order"], labels, params)
    assert decisions["A"]["accepted"] is True
    assert decisions["B+D"]["accepted"] is True
    assert decisions["B+D+E"]["accepted"] is True

    blocked = ls._stage_decisions(_row(market_regime="RISK_OFF"), summary["stage_order"], labels, params)
    for label in ("B", "B+D", "C", "D", "B+D+E"):
        assert blocked[label]["accepted"] is False
        assert blocked[label]["reason"] == "MARKET_REGIME_RISK_OFF"

    recent_report = ls._stage_decisions(_row(disclosure_periodic_days_since=6),
                                        summary["stage_order"], labels, params)
    assert recent_report["B+D"]["reason"] == "PERIODIC_REPORT_TOO_RECENT"
    assert recent_report["B+D+E"]["reason"] == "PERIODIC_REPORT_TOO_RECENT"


def test_earnings_abstain_does_not_reject_but_is_visible():
    summary = _summary()
    labels = summary["stage_labels"]
    params = {label: entry["params"] for label, entry in
              summary["cohorts"]["KOSPI"]["stage_params"].items()}
    row = _row(earnings_data_available=0, earnings_momentum=None, op_yoy_pct=None, opm_pct=None)
    decisions = ls._stage_decisions(row, summary["stage_order"], labels, params)
    # 감사 정책상 실적 미확보는 탈락이 아니라 보류(ABSTAIN)다.
    assert decisions["B+D+E"]["accepted"] is True
    rejected = ls._stage_decisions(_row(opm_pct=-4.0), summary["stage_order"], labels, params)
    assert rejected["B+D+E"]["accepted"] is False
    assert rejected["B+D+E"]["reason"] == "EARNINGS_THRESHOLD_NOT_MET"


def test_plan_levels_keep_stop_below_close_and_target_above():
    stop, take = ls._plan_levels(10000.0, 300.0, "trend_breakout")
    assert stop < 10000.0 <= take
    assert stop >= 10000.0 * 0.905 - 1e-6
    fallback_stop, fallback_take = ls._plan_levels(10000.0, 300.0, "unknown_variant")
    assert fallback_stop < 10000.0 < fallback_take


def test_status_reports_missing_and_written_files(tmp_path):
    missing = tmp_path / "none.json"
    assert ls.status(missing) == 1
    written = tmp_path / "live.json"
    written.write_text(json.dumps({"generated_at_kst": "2026-11-26 17:00:00+09:00",
                                   "signal_bar_date": "2026-11-26",
                                   "cohorts": {"KOSPI": {"counts_by_stage": {"A": 0},
                                                         "raw_signal_n": 0}},
                                   "notes": ["research only"]}), encoding="utf-8")
    assert ls.status(written) == 0


def test_market_cohort_mapping_matches_the_audit():
    assert ls.MARKET == {"KOSPI": "KR", "KOSDAQ": "KQ", "TIGER_ETF": "KR", "SOXL": "US"}
