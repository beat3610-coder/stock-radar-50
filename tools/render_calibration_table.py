"""Render the calibration-round comparison as a markdown table for the PR body."""
from __future__ import annotations

import json
import sys
from pathlib import Path

NAMES = ("development", "validation", "diagnostic")


def fmt(value, digits=2, suffix=""):
    if value is None:
        return "—"
    if isinstance(value, (int,)):
        return f"{value:,}"
    return f"{value:.{digits}f}{suffix}"


def row(label, base, new, key, digits=2, suffix="", delta=True):
    b, n = base.get(key), new.get(key)
    if isinstance(b, list):
        b = "/".join(str(x) for x in b)
        n = "/".join(str(x) for x in n)
        return f"| {label} | {b} | {n} | |"
    text = f"| {label} | {fmt(b, digits, suffix)} | {fmt(n, digits, suffix)} |"
    if delta and isinstance(b, (int, float)) and isinstance(n, (int, float)):
        text += f" {n - b:+.{digits}f}{suffix} |"
    else:
        text += " |"
    return text


def main(path):
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    for cohort, periods in report["cohorts"].items():
        if cohort == "selection":
            continue
        print(f"\n#### {cohort} — 63종목 고정 평가\n")
        print("| 지표 | main(PR #18) | 이번 라운드 | 차이 |")
        print("|---|---|---|---|")
        for period in NAMES:
            pair = periods[period]
            base, new = pair["base"], pair["new"]
            print(f"\n**{period}** (n={new.get('n')})\n")
            print(row("3방향 정확도 %", base, new, "accuracy_pct"), end="")
            print()
            print(row("기준선(최빈클래스) 대비 %p", base, new, "delta_vs_majority_pp"))
            print(row("인과 사전확률 정확도 %", base, new, "causal_prior_accuracy_pct", delta=False))
            print(row("Brier", base, new, "brier", digits=5, delta=True))
            print(row("Brier skill vs 기간 사전확률", base, new, "brier_skill_vs_prior", digits=5))
            print(row("Brier skill vs 인과 사전확률", base, new, "brier_skill_vs_causal", digits=5, delta=False))
            print(row("예측 분포 DOWN/FLAT/UP", base, new, "predicted_DOWN_FLAT_UP", delta=False))
            print(f"| collapse 상태 | {base.get('collapse') or '—'} | {new.get('collapse') or '—'} | |")
            print(f"| 원시 모델 argmax | — | {new.get('raw_argmax') or '—'} | |")
        selection = periods["selection"]
        print("\n**선택/보정 파라미터**\n")
        print("| 항목 | main | 이번 |")
        print("|---|---|---|")
        for key in ("selected_model_development_only", "selected_horizon_development_only",
                    "selected_development_datewise_spearman", "model_selection_status"):
            print(f"| {key} | {selection['base'].get(key)} | {selection['new'].get(key)} |")
    print("\n**B+D+E legacy 사다리 동일 여부:**", report.get("policy_ladder_equal"))


if __name__ == "__main__":
    main(sys.argv[1])
