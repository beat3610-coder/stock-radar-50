"""Conservative, *unvalidated* market-specific research filters.

The policy only downgrades BUY candidates. It cannot turn a SELL/HOLD into BUY.
No numerical parameter is represented as statistically calibrated.
All inputs come from the current/past bars supplied to engine.evaluate().
"""
import math

CANDIDATES = frozenset(("강한 매수 검토","매수 검토","기술매수 후보·수급 미확인"))
RESEARCH = "연구용 매수 후보·실거래 미승인"
HOLD = "위험관리·매수 유보"
POLICIES = {
    "KR": {"score":68, "min_rr":1.5, "min_upside":3.0, "max_stop_pct":8.0,
           "max_atr_pct":5.0, "min_net_pct":0.5, "rsi_max":69},
    "KQ": {"score":72, "min_rr":1.8, "min_upside":4.0, "max_stop_pct":7.0,
           "max_atr_pct":5.5, "min_net_pct":0.7, "rsi_max":68},
    "US": {"score":78, "min_rr":2.0, "min_upside":4.0, "max_stop_pct":6.0,
           "max_atr_pct":4.5, "min_net_pct":1.0, "rsi_max":67},
}

def _val(value):
    try:
        f=float(value)
        return f if math.isfinite(f) else None
    except (ValueError, TypeError):
        return None

def screen_candidate(signal, *, market, price, score, stop, target, plan, rsi,
                     atr, trend_ok, ma20, ma60, ma200,
                     previous_ma200=None, macdh=None, adx=None, pdi=None, mdi=None):
    """Return (display_signal, explanation, regime).

    No new BUY recommendation is possible: passing candidates remain research only.
    """
    if signal not in CANDIDATES:
        return signal, "기존 비매수 신호 유지", "해당 없음"
    p=POLICIES.get(market)
    if p is None:
        return HOLD, "미지원 시장", "불명"
    price=_val(price); score=_val(score); stop=_val(stop); target=_val(target)
    ma20=_val(ma20); ma60=_val(ma60); ma200=_val(ma200)
    older=_val(previous_ma200); rsi=_val(rsi); atr=_val(atr)
    macdh=_val(macdh); adx=_val(adx); pdi=_val(pdi); mdi=_val(mdi)
    reasons=[]
    if not all(v is not None for v in (price,score,stop,target,rsi,atr,ma20,ma60,ma200,older,macdh)):
        return HOLD, "가격·위험·추세 필수 데이터 누락", "불명"
    if price<=0 or not(0<stop<price<target):
        return HOLD, "손절가·익절가가 실제 가격 순서를 충족하지 않음", "불명"
    # Strict regime gate: sideways/bear = wait, never add risk for a higher score.
    bullish=(price>ma20>ma60>ma200 and ma200>older and bool(trend_ok))
    bearish=(price<ma200 or ma200<older)
    regime="상승" if bullish else ("하락" if bearish else "횡보")
    if not bullish: reasons.append(f"{regime} 국면·신규 매수 보류")
    if score<p["score"]:reasons.append(f"시장별 기술점수 {score:.0f}<{p['score']}")
    if rsi>p["rsi_max"] or rsi<45:reasons.append("RSI 과열 또는 모멘텀 부족")
    if macdh<=0:reasons.append("MACD 히스토그램 양수 미확인")
    if (adx is not None and pdi is not None and mdi is not None and
            adx>=25 and mdi>=pdi):
        reasons.append("ADX 하락 방향 우세")
    stop_pct=100*(price-stop)/price
    atr_pct=100*atr/price
    upside=100*(target-price)/price
    rr=(target-price)/(price-stop)
    if stop_pct>p["max_stop_pct"]:reasons.append(f"손절폭 {stop_pct:.2f}% 초과")
    if atr_pct>p["max_atr_pct"]:reasons.append(f"변동성 ATR {atr_pct:.2f}% 초과")
    if upside<p["min_upside"]:reasons.append(f"상승여력 {upside:.2f}% 부족")
    if rr<p["min_rr"]:reasons.append(f"손익비 {rr:.2f}R 부족")
    if not isinstance(plan,dict) or (plan.get("history_samples") or 0)<50:
        reasons.append("완료된 과거 경로 50개 미만")
    net=_val(plan.get("historical_net_expectancy_pct")) if isinstance(plan,dict) else None
    if net is None or net<p["min_net_pct"]:
        reasons.append("거래비용 반영 과거 평균 손익 미달")
    if reasons:
        return HOLD, " · ".join(reasons), regime
    return RESEARCH, ("시장별 위험 필터 통과, 그러나 독립 검증 미합격: 매수 권고 불가"
                      f" / 손절 {stop_pct:.1f}% / 기대손익비 {rr:.2f}R"), regime
