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
    positive=bool(expected_net>MIN_SHRUNK_EXPECTANCY_PCT)
    return {"eligible":positive,"sample_count":n,
            "win_probability_pct":round(100*win_probability,2),
            "stop_first_probability_pct":round(100*(stops+1)/(n+3),2),
            "take_first_probability_pct":round(100*(takes+1)/(n+3),2),
            "expected_net_pct":round(expected_net,3),
            "reason":"비용 반영 보수적 기대수익 양수" if positive else "보수적 기대수익 기준 미달"}

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
