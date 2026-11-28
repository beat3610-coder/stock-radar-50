"""Compare the calibration/decision round against the PR #18 (main) audit.

Usage:
    python tools/compare_calibration_round.py BASE.json NEW.json [--json OUT.json]

Reads only the direction3 blocks and the legacy policy ladder; every number is
copied from the reports, nothing is recomputed.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

COHORTS = ("KOSPI", "KOSDAQ")
PERIODS = ("development", "validation", "diagnostic")
FIT = "63_audit_symbols"


def _period_block(audit, cohort, period):
    direction = (audit.get("direction3") or {}).get(cohort) or {}
    comparison = (direction.get("universe_comparison") or {}).get(FIT) or {}
    return comparison.get(period) or {}


def _three_way(audit, cohort, period):
    return ((audit.get("direction3_three_way") or {}).get(cohort) or {}).get(period) or {}


def _counts(block):
    metrics = block.get("class_metrics") or {}
    return [int((metrics.get(name) or {}).get("predicted") or 0) for name in ("DOWN", "SIDEWAYS", "UP")]


def _round(value, digits=4):
    return None if value is None else round(float(value), digits)


def summarize(audit, cohort, period):
    three = _three_way(audit, cohort, period)
    block = _period_block(audit, cohort, period)
    metrics = block.get("metrics") or {}
    decision = metrics.get("decision") or {}
    row = {
        "n": three.get("evaluation_rows_n") or block.get("n"),
        "accuracy_pct": _round(three.get("accuracy_pct")),
        "majority_pct": _round(three.get("majority_class_baseline_accuracy_pct")),
        "delta_vs_majority_pp": _round(three.get("accuracy_delta_vs_majority_pp")),
        "causal_prior_accuracy_pct": _round(three.get("causal_prior_accuracy_pct")),
        "brier": _round(three.get("multiclass_brier"), 5),
        "prior_brier": _round(three.get("prior_brier"), 5),
        "brier_skill_vs_prior": _round(three.get("brier_skill_vs_period_prior"), 5),
        "brier_skill_vs_causal": _round(three.get("brier_skill_vs_causal_prior"), 5),
        "predicted_DOWN_FLAT_UP": _counts(three),
        "share_pct": {k: _round(v, 2) for k, v in (three.get("predicted_share_pct") or {}).items()},
        "collapse": three.get("decision_collapse_status"),
        "plain_argmax": (three.get("plain_argmax_decision") or {}).get("predicted_counts"),
        "raw_argmax": (three.get("raw_argmax_decision") or {}).get("predicted_counts"),
        "raw_argmax_accuracy_pct": _round((three.get("raw_argmax_decision") or {}).get("accuracy_pct")),
        "dispersion": three.get("probability_dispersion"),
        "decision_rule": decision.get("rule"),
        "decision_tau": decision.get("tau"),
    }
    calibration = (block.get("probability_sample") or {})
    row["probability_sample"] = calibration or None
    return row


def ladder(audit):
    """Legacy B+D+E buy ladder, which direction3 must not move."""
    out = {}
    ladder_block = audit.get("policy_ladder") or {}
    for market, value in ladder_block.items():
        if isinstance(value, dict):
            out[market] = {k: v for k, v in value.items() if isinstance(v, (int, float, str))}
    return out


def main(argv):
    base_path = Path(argv[1])
    new_path = Path(argv[2])
    out_path = Path(argv[4]) if len(argv) > 4 and argv[3] == "--json" else None
    base = json.loads(base_path.read_text(encoding="utf-8"))
    new = json.loads(new_path.read_text(encoding="utf-8"))
    report = {"base": str(base_path), "new": str(new_path), "cohorts": {}}
    for cohort in COHORTS:
        report["cohorts"][cohort] = {
            period: {"base": summarize(base, cohort, period), "new": summarize(new, cohort, period)}
            for period in PERIODS
        }
        direction_new = ((new.get("direction3") or {}).get(cohort) or {})
        direction_base = ((base.get("direction3") or {}).get(cohort) or {})
        report["cohorts"][cohort]["selection"] = {
            "base": {k: direction_base.get(k) for k in
                     ("selected_model_development_only", "selected_horizon_development_only",
                      "model_selection_status", "selected_development_datewise_spearman")},
            "new": {k: direction_new.get(k) for k in
                    ("selected_model_development_only", "selected_horizon_development_only",
                     "model_selection_status", "selected_development_datewise_spearman")},
            "new_calibration_by_universe": direction_new.get("calibration_by_universe")
            or direction_new.get("probability_calibration"),
        }
    report["policy_ladder_equal"] = ladder(base) == ladder(new)
    report["policy_ladder"] = {"base": ladder(base), "new": ladder(new)}
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if out_path:
        out_path.write_text(text, encoding="utf-8")
        print(f"wrote {out_path}")
    print(text)


if __name__ == "__main__":
    main(sys.argv)
