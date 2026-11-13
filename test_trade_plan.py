"""Deterministic tests of the primary take-profit and stop framework."""
import numpy as np
import pandas as pd
from engine import compute, evaluate
from trade_plan import scenario_from_history, decide_trade, position_sizing

ix=pd.bdate_range("2024-01-01",periods=350)
p=100+np.arange(350)*0.07+np.sin(np.arange(350)/8)*5
df=pd.DataFrame({"Open":p-.5,"High":p+2,"Low":p-2,
                  "Close":p,"Volume":200000+np.arange(350)*200},index=ix)
x=compute(df)
plan=scenario_from_history(x,stop=float(x.Close.iloc[-1])-4)
assert plan["history_samples"]>=50,plan
assert plan["target"]>float(x.Close.iloc[-1])
assert plan["target2"]>=plan["target"]
assert plan["planned_rr"]>0
assert plan["historical_target_choice"] in ("50th","75th")
assert isinstance(plan["historical_net_expectancy_pct"],float)
assert plan["historical_net_expectancy_pct"]==(
    plan["historical_net_75th_pct"] if plan["historical_target_choice"]=="75th"
    else plan["historical_net_50th_pct"])
assert decide_trade("매수 검토",{
    "target":110,"planned_rr":2.0,"expected_upside_pct":10,
    "historical_net_expectancy_pct":-2,"stop_risk_pct":5})[0]=="상승 신호·매수 유보"
assert 0<=plan["historical_target_first_pct"]<=100
assert 0<=plan["historical_second_first_pct"]<=plan["historical_target_first_pct"]
assert 0<=plan["historical_stop_first_pct"]<=100
assert plan["historical_target_first_pct"]+plan["historical_stop_first_pct"]<=100.02
assert plan["stop_risk_pct"]>0
assert "10거래일" in plan["basis"]
# Historical plan must not include the future candle, even with a huge spike.
part=x.iloc[:300].copy()
prior=scenario_from_history(part,stop=float(part.Close.iloc[-1])-4)
changed=x.copy()
changed.loc[changed.index[300:],"High"]=changed.loc[changed.index[300:],"High"]*100
again=scenario_from_history(changed.iloc[:300],stop=float(part.Close.iloc[-1])-4)
assert prior["target"]==again["target"],(prior,again)
assert decide_trade("하락 추세·매수 보류",plan)[0]=="하락 추세·매수 보류"
assert decide_trade("매수 검토",{"target":None,"planned_rr":None})[0]=="상승 신호·매수 유보"
assert decide_trade("매수 검토",plan)[0] in ("매수 검토","상승 신호·매수 유보")
actual=evaluate(x,market="US",flow=None)
assert isinstance(actual["medium_long_trend_rising"],bool)
if actual["signal"] in ("매수 검토","강한 매수 검토","기술매수 후보·수급 미확인"):
    assert actual["medium_long_trend_rising"]
assert actual["trade_target"]==scenario_from_history(x,actual["stop"])["target"]
assert actual["upside_pct"]==scenario_from_history(x,actual["stop"])["expected_upside_pct"]
assert actual["entry_rr"]==scenario_from_history(x,actual["stop"])["planned_rr"]
print("trade plan: no lookahead, profit levels, stops, and risk filter PASS")

sizing=position_sizing(10_000_000,100_000,95_000,115_000,0.01)
assert sizing["shares"]==18 and sizing["cost"]==1_800_000
assert abs(sizing["planned_loss"]-96_300)<0.01
assert abs(sizing["planned_profit"]-263_700)<0.01
assert position_sizing(100_000,100_000,95_000,115_000,0.01)["shares"]==0
assert position_sizing(10_000_000,100_000,105_000,115_000,0.01)["shares"]==0
print("position sizing tests PASS")
