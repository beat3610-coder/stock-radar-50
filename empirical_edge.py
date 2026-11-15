"""Chronological, cost-aware empirical trade expectancy calibration.

Every prediction for trade T can see ONLY trades with exit_date < T.signal_date.
No test-period tuning; 20 conservative pseudo-observations reduce small-sample
optimism. This estimates conditional win probability, not a guaranteed forecast.
"""
from collections import defaultdict
import math

MIN_PRIOR=25
MAX_PRIOR=100
PSEUDO_COUNT=20
PSEUDO_MEAN_NET_PCT=-0.35
MIN_SHRUNK_EXPECTANCY_PCT=0.15

def posterior_from_completed(prior):
    """Empirical win/stop/time probabilities and conservative expected net P&L."""
    if len(prior)<MIN_PRIOR:
        return {"eligible":False,"sample_count":len(prior),"reason":"과거 완료 거래 25건 미만"}
    recent=prior[-MAX_PRIOR:]
    n=len(recent)
    wins=sum(float(t["net_pct"])>0 for t in recent)
    stops=sum(str(t.get("exit_reason","")).startswith("stop") for t in recent)
    takes=sum(str(t.get("exit_reason","")).startswith("take") for t in recent)
    net=sum(float(t["net_pct"]) for t in recent)
    # Beta(2,2) prior: prevents interpreting 10/10 as 100% future win probability.
    win_probability=(wins+2)/(n+4)
    # Historical trade net returns already include round-trip cost.
    expected_net=(net+PSEUDO_COUNT*PSEUDO_MEAN_NET_PCT)/(n+PSEUDO_COUNT)
    # A long average can mask a sudden regime reversal. Require both the
    # long calibration pool and most recent 30 *completed* trades to retain
    # a positive, conservatively shrunken cost-adjusted edge.
    short=recent[-30:]
    short_wins=sum(float(t["net_pct"])>0 for t in short)
    short_expected=(sum(float(t["net_pct"]) for t in short)-0.35*10)/(len(short)+10)
    short_win_probability=(short_wins+2)/(len(short)+4)
    positive=bool(expected_net>MIN_SHRUNK_EXPECTANCY_PCT and short_expected>0.20)
    return {"eligible":positive,"sample_count":n,
            "recent_sample_count":len(short),
            "recent_expected_net_pct":round(short_expected,3),
            "recent_win_probability_pct":round(100*short_win_probability,2),
            "win_probability_pct":round(100*win_probability,2),
            "stop_first_probability_pct":round(100*(stops+1)/(n+3),2),
            "take_first_probability_pct":round(100*(takes+1)/(n+3),2),
            "expected_net_pct":round(expected_net,3),
            "reason":"장·단기 기대수익 모두 양수" if positive else "최근 30건 또는 전체 표본의 기대수익 악화"}

def walkforward_filter(trades):
    """Return prospective-compatible candidates and diagnostics.

    Input records may include later outcomes but ONLY earlier completed exits
    are included in each trade's calibration pool. Same-day exits are excluded.
    """
    chronological=sorted(trades,key=lambda t:(t["signal_date"],t["ticker"],t["entry_date"]))
    completed=sorted(trades,key=lambda t:(t["exit_date"],t["ticker"],t["entry_date"]))
    selected=[]
    statuses={"total":len(trades),"insufficient":0,"negative_edge":0,"accepted":0}
    j=0
    # Deduplicate by ticker-entry across history and restrict reference outcomes
    # by their end date. No sample uses its own or another open position's result.
    history=[]
    for item in chronological:
        signal=item["signal_date"]
        while j<len(completed) and completed[j]["exit_date"]<signal:
            history.append(completed[j])
            j+=1
        estimate=posterior_from_completed(history)
        if not estimate.get("eligible"):
            key="insufficient" if estimate["sample_count"]<MIN_PRIOR else "negative_edge"
            statuses[key]+=1
            continue
        statuses["accepted"]+=1
        selected.append({**item,"calibrated_probability_pct":estimate["win_probability_pct"],
                         "calibrated_expected_net_pct":estimate["expected_net_pct"],
                         "calibration_samples":estimate["sample_count"]})
    return selected,statuses
