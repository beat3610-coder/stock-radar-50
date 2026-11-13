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
    # At a historical entry price, test which barrier is reached FIRST.
    # Stop is assumed first when both prices touch in one daily bar.
    stop_fraction=risk/p
    first_hit=second_hit=stops_first=0
    for i in starts:
        first_outcome=second_outcome=None
        entry=close[i]
        for bar in range(i+1,i+horizon+1):
            hit_stop=lows[bar]<=entry*(1-stop_fraction)
            hit_first=highs[bar]>=entry*(1+q50)
            hit_second=highs[bar]>=entry*(1+q75)
            if first_outcome is None:
                if hit_stop:first_outcome="stop"
                elif hit_first:first_outcome="take"
            if second_outcome is None:
                if hit_stop:second_outcome="stop"
                elif hit_second:second_outcome="take"
            if first_outcome is not None and second_outcome is not None:
                break
        first_hit+=(first_outcome=="take")
        second_hit+=(second_outcome=="take")
        stops_first+=(first_outcome=="stop")
    hist_target_first_pct=round(100*first_hit/len(starts),2)
    hist_second_first_pct=round(100*second_hit/len(starts),2)
    hist_stop_first_pct=round(100*stops_first/len(starts),2)
    # Important: maximum future high ignores stop-before-target sequencing.
    # Compare actual realizable returns of the two historical exit barriers.
    def historical_net(target_fraction):
        result=[]
        for i in starts:
            entry=close[i]
            s=entry*(1-stop_fraction)
            take=entry*(1+target_fraction)
            price=close[i+horizon]
            for j in range(i+1,i+horizon+1):
                if float(x.Open.iloc[j])<=s:
                    price=float(x.Open.iloc[j]);break
                if float(x.Open.iloc[j])>=take:
                    price=float(x.Open.iloc[j]);break
                if lows[j]<=s:
                    price=s;break
                if highs[j]>=take:
                    price=take;break
            result.append(100*(price/entry-1)-0.35)
        return np.asarray(result,dtype=float)
    net50=historical_net(q50)
    net75=historical_net(q75)
    # Wider take profit needs 0.25 percentage point historic net advantage
    # over median target to avoid shifting exits on marginal differences.
    use75=bool(net75.mean()>=net50.mean()+0.25)
    net_expectancy=float(net75.mean() if use75 else net50.mean())
    target=p*(1+(q75 if use75 else q50))
    second=p*(1+q75)
    t1=target if target>p else None
    t2=second if second>p else None
    return {"target":t1,"target1":t1,"target2":t2,
            "expected_upside_pct":100*(q75 if use75 else q50),
            "planned_rr":(t1-p)/risk if t1 else None,
            "historical_net_expectancy_pct":round(net_expectancy,3),
            "historical_target_choice":"75th" if use75 else "50th",
            "historical_net_50th_pct":round(float(net50.mean()),3),
            "historical_net_75th_pct":round(float(net75.mean()),3),
            "history_samples":len(starts),
            "positive_rate_pct":round(100*np.mean(favorable>0),2),
            "downside_p75_pct":round(100*np.quantile(adverse,0.75),2),
            "stop_risk_pct":round(100*risk/p,2),
            "historical_target_first_pct":hist_target_first_pct,
            "historical_second_first_pct":hist_second_first_pct,
            "historical_stop_first_pct":hist_stop_first_pct,
            "basis":f"과거 완료된 {horizon}거래일 경로 {len(starts)}개 / 중앙값·75백분위 목표의 손절 우선·왕복비용 포함 손익 비교 (선택: {'75백분위' if use75 else '중앙값'}, 과거 평균 {net_expectancy:+.2f}%; 미래 수익 확률 아님)"}

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
    net=plan.get("historical_net_expectancy_pct")
    if net is None or net<=0:
        reasons.append("완료된 과거 경로의 손절 선행·왕복비용 반영 평균 손익이 0% 이하")
    if plan["stop_risk_pct"] is not None and plan["stop_risk_pct"]>max_risk_pct:
        reasons.append(f"손절폭 {plan['stop_risk_pct']:.2f}% > {max_risk_pct:.1f}%")
    if reasons:return hold," · ".join(reasons)
    return signal,("과거 중앙 상승여력 "
                   f"{plan['expected_upside_pct']:.2f}%, 손익비 {plan['planned_rr']:.2f}R "
                   f"(표본 {plan['history_samples']}; 미래 예측 아님)")

def position_sizing(capital, entry, stop, take, risk_fraction=0.01,
                    round_trip_cost_fraction=0.0035):
    """Whole-share allocation with assumed fees/slippage on stop and take.

    No gap guarantee: actual loss can still exceed the estimated plan.
    """
    valid=(all(np.isfinite(x) for x in
                (capital,entry,stop,risk_fraction,round_trip_cost_fraction))
           and capital>0 and entry>0 and 0<stop<entry and
           0<risk_fraction<=0.05 and 0<=round_trip_cost_fraction<=0.05)
    if not valid:
        return {"shares":0,"cost":0,"planned_loss":None,"planned_profit":None}
    per_share_loss=(entry-stop)+entry*round_trip_cost_fraction
    shares=min(int(capital//(entry*(1+round_trip_cost_fraction/2))),
               int((capital*risk_fraction)//per_share_loss))
    shares=max(0,shares)
    net_profit=(take-entry)-entry*round_trip_cost_fraction if (
        take is not None and np.isfinite(take) and take>entry) else None
    return {"shares":shares,"cost":shares*entry,
            "planned_loss":shares*per_share_loss,
            "planned_profit":shares*net_profit if net_profit is not None else None}
