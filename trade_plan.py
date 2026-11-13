"""Risk-first exit planning, with strictly historical empirical excursion levels.

All prices are daily-bar scenarios, not guaranteed fills or forecasts.
Forward excursion calibration *at time t* uses samples whose full future
10-session window ended before t; no future data leak.
"""
import numpy as np
import pandas as pd

HORIZON=10
MIN_SAMPLES=50

def scenario_from_history(x, stop, horizon=HORIZON, min_samples=MIN_SAMPLES):
    """Return targets and downside context calibrated on past completed windows."""
    p=float(x.Close.iloc[-1])
    if stop is None or not np.isfinite(stop) or not 0<stop<p or len(x)<200:
        return {"target":None,"target1":None,"target2":None,
                "expected_upside_pct":None,"planned_rr":None,
                "history_samples":0,"positive_rate_pct":None,
                "downside_p75_pct":None,"stop_risk_pct":None,
                "basis":"손절가 또는 200거래일 자료 부족"}
    close=x.Close.to_numpy(dtype=float)
    highs=x.High.to_numpy(dtype=float)
    lows=x.Low.to_numpy(dtype=float)
    ma20=x.MA20.to_numpy(dtype=float)
    ma60=x.MA60.to_numpy(dtype=float)
    # Historical entry-day data must have an entirely observed future window.
    indices=np.arange(max(190,len(x)-252),len(x)-horizon)
    mask=np.isfinite(ma20[indices]) & np.isfinite(ma60[indices])
    if np.isfinite(x.MA20.iloc[-1]) and np.isfinite(x.MA60.iloc[-1]):
        # Use comparable historical up/downtrend regimes when there are enough observations.
        trending=bool(x.MA20.iloc[-1]>=x.MA60.iloc[-1])
        similar=mask & ((ma20[indices]>=ma60[indices])==trending)
        if similar.sum()>=min_samples:
            mask=similar
    starts=indices[mask]
    if len(starts)<min_samples:
        return {"target":None,"target1":None,"target2":None,
                "expected_upside_pct":None,"planned_rr":None,
                "history_samples":int(len(starts)),"positive_rate_pct":None,
                "downside_p75_pct":None,"stop_risk_pct":round(100*(p-stop)/p,2),
                "basis":"완료된 과거 10거래일 경로 부족"}
    favorable=np.array([highs[i+1:i+horizon+1].max()/close[i]-1 for i in starts])
    adverse=np.array([1-lows[i+1:i+horizon+1].min()/close[i] for i in starts])
    # Median opportunity as primary reference; 75th quantile as optimistic case.
    q50=max(0,float(np.quantile(favorable,0.50)))
    q75=max(q50,float(np.quantile(favorable,0.75)))
    risk=p-stop
    target=p*(1+q50)
    second=p*(1+q75)
    t1=target if target>p else None
    t2=second if second>p else None
    return {"target":t1,"target1":t1,"target2":t2,
            "expected_upside_pct":100*q50,
            "planned_rr":(t1-p)/risk if t1 else None,
            "history_samples":len(starts),
            "positive_rate_pct":round(100*np.mean(favorable>0),2),
            "downside_p75_pct":round(100*np.quantile(adverse,0.75),2),
            "stop_risk_pct":round(100*risk/p,2),
            "basis":f"과거 같은 추세의 완료된 {horizon}거래일 경로 {len(starts)}개 / 유리변동 중앙값·75백분위 (미래 수익 확률 아님)"}

def decide_trade(signal, plan, score=None, min_rr=1.2, min_upside=2.0,
                 max_risk_pct=12.0):
    """Buy candidate only if historical scenario is favorable versus proposed stop.

    Does not promote a technical risk warning to BUY. The numeric thresholds are
    configurable heuristics and must be validated out of sample before sale.
    """
    if signal not in ("매수 검토","강한 매수 검토","기술매수 후보·수급 미확인"):
        return signal,"기술적 추세·위험 경고 우선"
    unconfirmed=(signal=="기술매수 후보·수급 미확인")
    hold="기술상승·매수 유보(수급 미확인)" if unconfirmed else "상승 신호·매수 유보"
    if plan["target"] is None or plan["planned_rr"] is None:
        return hold,"역사적 상승여력·손절가 산정에 필요한 경로 부족"
    reasons=[]
    if plan["expected_upside_pct"]<min_upside:
        reasons.append(f"과거 중앙 상승여력 {plan['expected_upside_pct']:.2f}% < {min_upside:.1f}%")
    if plan["planned_rr"]<min_rr:
        reasons.append(f"경험 기반 기대손익비 {plan['planned_rr']:.2f}R < {min_rr:.1f}R")
    if plan["stop_risk_pct"] is not None and plan["stop_risk_pct"]>max_risk_pct:
        reasons.append(f"손절폭 {plan['stop_risk_pct']:.2f}% > {max_risk_pct:.1f}%")
    if reasons:return hold," · ".join(reasons)
    return signal,("과거 중앙 상승여력 "
                   f"{plan['expected_upside_pct']:.2f}%, 손익비 {plan['planned_rr']:.2f}R "
                   f"(표본 {plan['history_samples']}; 미래 예측 아님)")

def position_sizing(capital, entry, stop, take, risk_fraction=0.01):
    """Whole-share position size with max loss at proposed stop (no gaps/fees)."""
    valid=(all(np.isfinite(x) for x in (capital,entry,stop,risk_fraction))
           and capital>0 and entry>0 and 0<stop<entry
           and 0<risk_fraction<=0.05)
    if not valid:
        return {"shares":0,"cost":0,"planned_loss":None,"planned_profit":None}
    shares=min(int(capital//entry),int((capital*risk_fraction)//(entry-stop)))
    shares=max(0,shares)
    return {"shares":shares,"cost":shares*entry,
            "planned_loss":shares*(entry-stop),
            "planned_profit":shares*(take-entry) if take is not None and np.isfinite(take)
                 and take>entry else None}
