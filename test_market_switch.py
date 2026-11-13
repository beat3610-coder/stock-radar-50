"""Streamlit market-switch smoke test using isolated stock universes (no network)."""
import json
from pathlib import Path
import universe
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
Path("data").mkdir(exist_ok=True)
Path("data/market-switch-test.json").write_text(
    json.dumps({"success":True,"screens":results},ensure_ascii=False,indent=2),encoding="utf8")
print(json.dumps({"success":True,"screens":results},ensure_ascii=False))
