"""Streamlit market-switch smoke test using isolated stock universes (no network)."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import universe
import providers
from streamlit.testing.v1 import AppTest

def fake_universe():
    return {
      "KR":[("005930","KR_SAMPLE_SAMSUNG","KR","Tech")],
      "KQ":[("196170","KQ_SAMPLE_ALTEOGEN","KQ","Bio")],
      "US":[("NVDA","US_SAMPLE_NVIDIA","US","Semis")],
      "KR_source":"test KR","KQ_source":"test KQ","US_source":"test US",
      "KR_error":None,"KQ_error":None,"KR_index_count":1,
      "KQ_index_count":1,"US_index_count":1,"US_cap_sorted":True
    }

universe.get_universe=fake_universe
days=pd.bdate_range("2025-01-01",periods=350)
close=100+np.arange(350)*0.07+np.sin(np.arange(350)/8)*4
mock_prices=pd.DataFrame({"Open":close-.1,"High":close+1.5,
    "Low":close-1.5,"Close":close,"Volume":200000+np.arange(350)*100},index=days)
providers.get_prices=lambda code,market:(mock_prices.copy(),"TEST:MOCK")
providers.get_flows=lambda code,date:{"date":date,"외국인":{"5일":10,"20일":20},
    "기관":{"5일":20,"20일":30}}
providers.verify_recent_close=lambda *args,**kwargs:{
    "status":"agree","difference_pct":0.0,"reason":"fixture","secondary_close":None}
at=AppTest.from_file("app.py",default_timeout=25).run()
def check(market, sector):
    assert not at.exception, [str(x.message) for x in at.exception]
    titles=[x.value for x in at.subheader]
    # Ranking subheader must reflect *one* market. Its other-market titles must not exist.
    assert any(sector in x for x in titles), titles
    assert len([x for x in titles if "시총" in x or "Nasdaq" in x])==1, titles
    detail=[x for x in at.selectbox if x.label=="종목 검색 · 선택"]
    assert len(detail)==1, [x.label for x in at.selectbox]
    values=list(detail[0].options)
    assert len(values)==1,values
    assert market in values[0],(sector,values)
    return {"market":sector,"ranking_titles":titles,"detail_options":values}

results=[]
results.append(check("KR_SAMPLE","코스피"))
market_control=[x for x in at.get("segmented_control") if x.label=="분석할 시장 선택"]
assert len(market_control)==1
at=market_control[0].set_value("KQ").run()
results.append(check("KQ_SAMPLE","코스닥"))
market_control=[x for x in at.get("segmented_control") if x.label=="분석할 시장 선택"]
at=market_control[0].set_value("US").run()
results.append(check("US_SAMPLE","Nasdaq"))
market_control=[x for x in at.get("segmented_control") if x.label=="분석할 시장 선택"]
at=market_control[0].set_value("KR").run()
results.append(check("KR_SAMPLE","코스피"))
# Simulate a customer clicking a detailed chart after market switching;
# this exercises the actual take-profit/stop/risk sizing UI code.
get_chart=[b for b in at.button if b.label=="📈 차트 및 매매 판단 조회"]
assert len(get_chart)==1
at=get_chart[0].click().run()
assert not at.exception, [str(x.message) for x in at.exception]
details=[x.value for x in at.subheader]
assert any("익절·손절" in title for title in details), details
Path("data").mkdir(exist_ok=True)
Path("data/market-switch-test.json").write_text(
    json.dumps({"success":True,"chart_and_risk_panel":True,"screens":results},ensure_ascii=False,indent=2),encoding="utf8")
print(json.dumps({"success":True,"screens":results},ensure_ascii=False))
