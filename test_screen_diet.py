"""화면 다이어트 회귀 테스트 (Streamlit AppTest, 네트워크 없음).

지키려는 것:
  1. 결과표·후보표의 기본 표시는 핵심 9개 열뿐이고, 나머지 진단 열은 기본으로 숨는다.
  2. 숨긴 열은 체크박스로만 펼치고, 계산·CSV용 내부 데이터(74개 열)는 그대로 남는다.
  3. 상단 연구·검증 패널은 접힌 채로 뜨고, B+D+E 패널은 접힌 상태에서도
     오늘 후보 수와 시장 국면을 한 줄로 보여준다.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

import universe
import providers
from streamlit.testing.v1 import AppTest

CORE = ["구분", "종목", "매매 판단", "종가", "전일대비 %",
        "통합 목표가", "손절 참고", "기대손익비(R)", "점수"]

universe.get_universe = lambda: {
    "KR": [("005930", "KR_SAMPLE_SAMSUNG", "KR", "Tech")], "KQ": [], "US": [],
    "KR_source": "test KR", "KQ_source": "test KQ", "US_source": "test US",
    "KR_error": None, "KQ_error": None, "KR_index_count": 1,
    "KQ_index_count": 0, "US_index_count": 0, "US_cap_sorted": True}
days = pd.bdate_range("2025-01-01", periods=350)
close = 100 + np.arange(350) * 0.07 + np.sin(np.arange(350) / 8) * 4
mock_prices = pd.DataFrame({"Open": close - .1, "High": close + 1.5,
                            "Low": close - 1.5, "Close": close,
                            "Volume": 200000 + np.arange(350) * 100}, index=days)
providers.get_prices = lambda code, market: (mock_prices.copy(), "TEST:MOCK")
providers.get_flows = lambda code, date: {"date": date, "외국인": {"5일": 10, "20일": 20},
                                          "기관": {"5일": 20, "20일": 30}}
providers.verify_recent_close = lambda *args, **kwargs: {
    "status": "agree", "difference_pct": 0.0, "reason": "fixture", "secondary_close": None}


def tables_with_core(state):
    return [d.value for d in state.dataframe
            if list(getattr(d.value, "columns", []))[:len(CORE)] == CORE]


at = AppTest.from_file("app.py", default_timeout=40).run()
assert not at.exception, [str(x.message) for x in at.exception]

# 3) 상단 연구·검증 패널은 모두 접힌 상태로 시작한다.
research_panels = ["🧭 B+D+E 실전 필터 · 최근 후보", "🔬 예측모델 개선 현황",
                   "🧪 실전 투자 합격 검증표", "🔍 독립 검증·판매 적합성",
                   "🧪 3회 확률 개선 연구", "🎯 매수 성공률 60% 목표",
                   "📊 시장별 매매전략 비교", "🛠️ 검증·승인·지속학습 센터"]
open_panels = []
for panel in research_panels:
    matches = [e for e in at.expander if e.label.startswith(panel)]
    assert len(matches) == 1, (panel, [e.label for e in at.expander])
    if getattr(matches[0].proto, "expanded", False):
        open_panels.append(panel)
assert not open_panels, open_panels

# B+D+E 패널은 접힌 채로도 오늘 후보 수·시장 국면을 한 줄로 보여준다.
bde = [e for e in at.expander if e.label.startswith("🧭 B+D+E")][0].label
assert "오늘 B+D+E 후보" in bde, bde
assert "시장국면" in bde, bde

# 분석을 실행해 결과표를 만든다.
run = [b for b in at.button if b.label == "▶ 다음 50종목 실제 시세 분석"]
assert len(run) == 1, [b.label for b in at.button][:10]
at = run[0].click().run()
assert not at.exception, [str(x.message) for x in at.exception]

# 1) 기본 표시 열 = 핵심 9개 (결과표 + 후보표 둘 다)
shown = tables_with_core(at)
assert len(shown) == 2, [list(getattr(d.value, "columns", []))[:5] for d in at.dataframe]
for frame in shown:
    assert list(frame.columns) == CORE, list(frame.columns)
# 빈값이 'None' 글자로 그려지지 않도록 숫자 열은 숫자형으로 그린다(원본 데이터는 그대로).
assert str(shown[0]["통합 목표가"].dtype).startswith("float"), shown[0]["통합 목표가"].dtype
assert str(shown[0]["전일대비 %"].dtype).startswith("float"), shown[0]["전일대비 %"].dtype
assert shown[0]["매매 판단"].dtype == object, shown[0]["매매 판단"].dtype

# 2) 상세 열은 기본으로 숨고, 체크박스로 펼친다.
detail_boxes = [c for c in at.checkbox if c.label.startswith("🔧 상세 진단 열")]
assert len(detail_boxes) == 1, [c.label for c in at.checkbox]
assert detail_boxes[0].value is False, detail_boxes[0].value
at = detail_boxes[0].set_value(True).run()
assert not at.exception, [str(x.message) for x in at.exception]
expanded = [d.value for d in at.dataframe
            if "매물대 분석" in list(getattr(d.value, "columns", []))]
assert len(expanded) >= 1, "상세 체크 후에도 상세 열이 보이지 않음"
assert list(expanded[0].columns)[:len(CORE)] == CORE, list(expanded[0].columns)[:12]
assert len(expanded[0].columns) > 50, len(expanded[0].columns)

cand_boxes = [c for c in at.checkbox if c.label.startswith("🔧 후보 상세 열")]
assert len(cand_boxes) == 1, [c.label for c in at.checkbox]
assert cand_boxes[0].value is False, cand_boxes[0].value
at = cand_boxes[0].set_value(True).run()
assert not at.exception, [str(x.message) for x in at.exception]

# 2-2) 계산·CSV용 내부 데이터는 그대로: 세션의 종목 행은 70개가 넘는 열을 유지한다.
rows = at.session_state["results_v26_KR_flow"]["rows"]
assert rows, "분석 결과가 세션에 없음"
first_row = list(rows.values())[0]
for column in ("매물대 분석", "시세 출처", "종가 교차검증", "20일선", "200일선",
               "RSI", "과거 전체 표본", "익절 산정 근거", "매매 판단 근거"):
    assert column in first_row, column
assert len(first_row) > 60, len(first_row)

Path("data").mkdir(exist_ok=True)
result = {"success": True, "core_columns": CORE,
          "collapsed_research_panels": research_panels,
          "bde_headline": bde,
          "default_columns": len(CORE),
          "expanded_columns": len(expanded[0].columns),
          "internal_row_columns": len(first_row)}
Path("data/screen-diet-test.json").write_text(
    json.dumps(result, ensure_ascii=False, indent=2), encoding="utf8")
print(json.dumps(result, ensure_ascii=False))
