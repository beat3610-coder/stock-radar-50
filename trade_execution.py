"""Shared decision-time/execution-time plan math (research only).

All prices used at signal close are observed prices. Next-session entry must
pass the identical gate using its *actual* open, or the order is cancelled.
"""
import math

COST_PCT = 0.35
MAX_GAP_PCT = 2.5
MAX_STOP_PCT = 7.0
MIN_NET_UPSIDE_PCT = 3.0
MIN_NET_RR = 1.30

def entry_quality(entry,stop,target,reference=None,cost_pct=COST_PCT):
    try:
        entry,stop,target,cost_pct=map(float,(entry,stop,target,cost_pct))
        if not all(math.isfinite(v) for v in (entry,stop,target,cost_pct)):
            raise ValueError("nonfinite")
        if not (0<stop<entry<target and 0<=cost_pct<=5):
            raise ValueError("invalid price levels")
        if reference is not None:
            reference=float(reference)
            if not math.isfinite(reference) or reference<=0:
                raise ValueError("invalid reference")
    except (TypeError,ValueError,OverflowError):
        return {"accepted":False,"reason":"가격·비용 입력 오류",
                "net_rr":None,"net_upside_pct":None,"stop_risk_pct":None}
    gap_pct=100*(entry/reference-1) if reference is not None else 0.0
    risk=entry-stop
    cost=entry*cost_pct/100
    reward=target-entry-cost
    net_rr=reward/(risk+cost)
    stop_pct=100*risk/entry
    upside_pct=100*reward/entry
    reasons=[]
    if abs(gap_pct)>MAX_GAP_PCT: reasons.append("허용 시가 갭 초과")
    if stop_pct>MAX_STOP_PCT: reasons.append("손절폭 7% 초과")
    if upside_pct<MIN_NET_UPSIDE_PCT: reasons.append("비용 반영 상승여력 3% 미만")
    if net_rr<MIN_NET_RR: reasons.append("실제 진입 손익비 1.30R 미만")
    return {"accepted":not reasons,
            "reason":"기준 충족" if not reasons else " · ".join(reasons),
            "net_rr":round(net_rr,3),
            "net_upside_pct":round(upside_pct,3),
            "stop_risk_pct":round(stop_pct,3),
            "gap_pct":round(gap_pct,3)}
