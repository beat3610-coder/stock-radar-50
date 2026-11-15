"""Executable research playbook from the same signal logic used in strategy_lab.
Returns today's close-time hypothesis only. NEVER uses tomorrow's open or high.
"""
from math import isfinite
from strategy_lab import PARAMS,qualifying
from trade_plan import position_sizing
from trade_execution import entry_quality, MAX_GAP_PCT

def playbook(indicators,market,capital=10_000_000.0):
    if indicators is None or len(indicators)<220:
        return {"date":None,"setups":[],"reason":"At least 220 daily bars required"}
    r=indicators.iloc[-1];p=indicators.iloc[-2]
    close=float(r.Close);atr=float(r.ATR)
    if not(isfinite(close) and close>0 and isfinite(atr) and atr>0):
        return {"date":None,"setups":[],"reason":"Incomplete latest close/ATR"}
    prior_high=float(indicators.High.iloc[-21:-1].max())
    setups=[]
    for name,params in PARAMS.items():
        if not qualifying(r,p,market,name,prior_high):
            continue
        stop=max(close*(1-0.095),close-params["stop_atr"]*atr)
        stop=min(stop,close*0.98)
        take=close+params["rr"]*(close-stop)
        check=entry_quality(close,stop,take,reference=close)
        if not check["accepted"]:continue
        rr=(take-close)/(close-stop)
        sized=position_sizing(capital,close,stop,take,
                              risk_fraction=.01,max_allocation_fraction=.20)
        if sized["shares"]<=0:continue
        setups.append({
            "strategy":name,"signal_price":round(close,4),
            "entry_reference":round(close,4),
            "entry_min":round(close*0.975,4),
            "entry_max":round(close*1.025,4),
            "stop":round(stop,4),"target":round(take,4),
            "stop_risk_pct":round(100*(close-stop)/close,2),
            "target_upside_pct":round(100*(take-close)/close,2),
            "risk_reward":round(rr,2),
            "net_reward_risk":check["net_rr"],
            "cost_adjusted_upside_pct":check["net_upside_pct"],
            "open_recheck_required":True,
            "max_shares":sized["shares"],
            "capital_used":round(sized["cost"],2),
            "planned_loss":round(sized["planned_loss"],2),
            "planned_profit":round(sized["planned_profit"],2),
            "status":"RESEARCH_ONLY",
            "rules":"연구용: 익일 시가가 기준종가 ±2.5% 이내여야 하며, 동일 진입 조건을 실제 시가로 다시 통과해야 유효합니다."})
    return {"date":str(indicators.index[-1].date()),"setups":setups,
            "reason":None if setups else "No researched entry setup on the latest completed daily bar",
            "currency":"KRW" if market in ("KR","KQ") else "USD",
            "live_approved":False}
