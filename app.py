import json
from pathlib import Path
from access_control import enforce_access, access_mode
from trade_plan import position_sizing
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from universe import get_universe
from engine import compute,evaluate
from providers import get_prices,get_flows,verify_recent_close

import audit_runtime
audit_runtime.ensure_started()

st.set_page_config(page_title="STOCK RADAR | 투자분석",page_icon="📈",layout="wide")
# Gate before loading universe or any third-party data for private customers.
enforce_access(st)
st.markdown("<style>.block-container{padding-top:1.4rem;max-width:1560px}.stMetric{padding:12px;border:1px solid #64748b44;border-radius:12px}</style>",unsafe_allow_html=True)
st.title("📈 STOCK RADAR")
st.error("⚠️ 현재 전략은 독립 수익 검증 미통과 상태입니다. 표시되는 매수 후보는 연구용이며 실전 매수 권고가 아닙니다.")
if access_mode()!="private":
    st.caption("공개 베타 · 비상업용 평가 버전 | 공식 시세 라이선스·결제·회원 운영 미연결")
else:
    st.caption("접근 인증 적용 · 초대 회원 전용 | 결제 상태는 외부 운영자가 별도로 관리")
    with st.sidebar:
        st.caption("로그인 계정: "+str(st.user.get("email") or "—"))
        st.button("로그아웃",on_click=st.logout,key="logout")
st.caption("🇰🇷 코스피 시총 100 + TIGER ETF 2개 | 🇰🇷 코스닥 시총 100 | 🇺🇸 Nasdaq-100 기업 100 + SOXL | 종가 기준 분석")
st.caption("매매 계획 중심: 1차·2차 익절 참고, 손절 참고, 위험폭과 손익비를 함께 확인합니다. "
           "익절가는 과거 가격 경로의 통계값이며 미래 수익 보장이나 확정 목표가가 아닙니다. 실제 주문 전에 공식 시세를 확인하세요.")

@st.cache_data(ttl=86400,show_spinner=False)
def universe():return get_universe()

@st.cache_data(ttl=21600,show_spinner=False)
def priced(code,market):
    d,source=get_prices(code,market)
    return compute(d),source

@st.cache_data(ttl=10800,show_spinner=False)
def flows(code,date):return get_flows(code,date)

@st.cache_data(ttl=21600,show_spinner=False)
def quote_check(code,market,date,close,source):
    return verify_recent_close(code,market,date,close,source)

try: listing=universe()
except Exception as e:
    st.error(f"종목 목록 초기화 실패: {e}")
    st.stop()

def analyze_item(item,with_flow=True):
    code,name,market,sector=item
    data,src=priced(code,market)
    flow_data=None
    flow_error=None
    if market in ("KR","KQ") and with_flow:
        try:
            flow_data=flows(code,str(data.index[-1].date()))
        except Exception as exc:
            flow_error=str(exc)[:160]
    r=evaluate(data,market=market,flow=flow_data)
    qc=quote_check(code,market,str(data.index[-1].date()),float(data.Close.iloc[-1]),src)
    r["quote_check"]=qc
    # A modest secondary-feed closing discrepancy is a review flag, not the
    # economic entry criterion; extremely divergent prices still fail closed.
    if qc["status"]=="mismatch" and qc["difference_pct"] is not None and qc["difference_pct"]>2.0:
        r["signal"]="시세 큰 차이·매수 보류"
        r["entry_reason"]=qc["reason"]
    return data,src,r,flow_data,flow_error

def to_row(item,with_flow=True,market_rank=None):
    code,name,market,sector=item
    data,src,r,flow_data,flow_error=analyze_item(item,with_flow=with_flow)
    f=flow_data or {}
    foreign=f.get("외국인",{})
    institution=f.get("기관",{})
    return {
      "구분":(f"코스피 {market_rank}위" if market_rank is not None else "국내 ETF") if market=="KR" else (f"코스닥 {market_rank}위" if market_rank is not None else "코스닥") if market=="KQ" else (f"나스닥 {market_rank}위" if market_rank is not None else "미국 ETF"),
      "종목":name,"코드":code,"업종":sector,
      "거래일":r["date"],"통화":"KRW" if market in ("KR","KQ") else "USD",
      "종가":round(r["price"],2),"전일대비 %":round(r["change"],2),
      "점수":r["score"],"점수 기준":r["score_basis"],"기술점수":r["technical_score"],"기술만점":r["technical_max"],
      "매매 판단":r["signal"],
      "시장 위험국면":r.get("risk_regime","미확인"),
      "실전 전략 승인":"미승인" if not r.get("live_strategy_approved") else "승인",
      "기술 판단(목표 검증 전)":r["technical_signal_before_gate"],
      "목표 상승여력 %":round(r["upside_pct"],2) if r["upside_pct"] is not None else None,
      "기대손익비(R)":round(r["entry_rr"],2) if r["entry_rr"] is not None else None,
      "매매 판단 근거":r["entry_reason"],
      "중장기 상승추세":("확인" if r["medium_long_trend_rising"] else "미확인"),
      "외국인 5일 순매수(주)":foreign.get("5일"),"외국인 20일 순매수(주)":foreign.get("20일"),
      "기관 5일 순매수(주)":institution.get("5일"),"기관 20일 순매수(주)":institution.get("20일"),
      "RSI":round(r["rsi"],1) if r["rsi"] is not None else None,
      "RSI 5일 변화":round(r["rsi_delta5"],1) if r["rsi_delta5"] is not None else None,
      "ADX":round(r["adx"],1) if r["adx"] is not None else None,
      "20일선":round(r["ma20"],2) if r["ma20"] is not None else None,
      "60일선":round(r["ma60"],2) if r["ma60"] is not None else None,
      "120일선":round(r["ma120"],2) if r["ma120"] is not None else None,
      "200일선":round(r["ma200"],2) if r["ma200"] is not None else None,
      "거래량/20일 평균":round(r["volratio"],2) if r["volratio"] is not None else None,
      "OBV":"매수 우세" if r["obv_up"] else "우세 아님" if r["volume_available"] else "자료 없음",
      "52주 고점":round(r["hi52"],2) if r["hi52"] else None,
      "52주 저점":round(r["lo52"],2) if r["lo52"] else None,
      "손절 참고":round(r["stop"],2) if r["stop"] is not None else None,
      "손절폭 %":round(r["stop_risk_pct"],2) if r["stop_risk_pct"] is not None else None,
      "통합 목표가":round(r["trade_target"],2) if r["trade_target"] is not None else None,
      "2차 익절 참고":round(r["trade_target2"],2) if r["trade_target2"] is not None else None,
      "과거 저항 참고":round(r["integrated_target"],2) if r["integrated_target"] is not None else None,
      "과거 경로 표본":r["historical_samples"],
      "과거 비용 반영 손익 %":r["historical_net_expectancy_pct"],
      "선택 익절 기준":r["historical_target_choice"],
      "과거 1차 익절 우선 %":r["historical_target_first_pct"],
      "과거 2차 익절 우선 %":r["historical_second_first_pct"],
      "과거 손절 우선 %":r["historical_stop_first_pct"],
      "익절 산정 근거":r["trade_plan_basis"],
      "목표 여건":r["target_grade"],
      "목표 분석점수":r["target_signal_points"],
      "목표 수급반영":r["target_inputs"],
      "거래량 돌파":("확인" if r["resistance_breakout"] else "미확인") if r["resistance_breakout"] is not None else "자료 부족",
      "1차 저항 목표":round(r["target1"],2) if r["target1"] is not None else None,
      "2차 저항 목표":round(r["target2"],2) if r["target2"] is not None else None,
      "2R 손익비 목표":round(r["target"],2) if r["target"] is not None else None,
      "1차 저항 근거":r["target1_basis"],
      "2차 저항 근거":r["target2_basis"],
      "1차 기대손익비(R)":round(r["target1_r"],2) if r["target1_r"] is not None else None,
      "2차 기대손익비(R)":round(r["target2_r"],2) if r["target2_r"] is not None else None,
      "통합목표 손익비(R)":round(r["target_r"],2) if r["target_r"] is not None else None,
      "목표 판단 근거":r["target_evidence"],
      "매물대 분석":r["volume_profile"],
      "시세 출처":src,
      "종가 교차검증":("보조 시세 일치(공식값 미검증)" if r["quote_check"]["status"]=="agree"
                       else "종가 차이 큼·매수 보류" if r["quote_check"]["status"]=="mismatch"
                            and r["quote_check"]["difference_pct"] is not None
                            and r["quote_check"]["difference_pct"]>2.0
                       else "종가 소폭 차이·참고" if r["quote_check"]["status"]=="mismatch"
                       else "교차검증 불가"),
      "종가 차이 %":r["quote_check"]["difference_pct"],
      "교차검증 설명":r["quote_check"]["reason"]}

def fmt(v,market="KR"):
    return "—" if v is None else (f"{v:,.2f}" if market=="US" else f"{v:,.0f}")

with st.sidebar:
    st.header("조회 설정")
    st.info("코스피 100 + 코스닥 100 + 나스닥100을 시장별로 분석합니다. 필수 ETF 3개는 기존 시장에 추가됩니다.")
    st.caption("시장 전환은 메인 화면의 시장 선택 버튼으로 합니다. 결과표와 상세 차트가 함께 전환됩니다.")
    if st.button("🔄 종목·시세 갱신",use_container_width=True):
        st.cache_data.clear()
        for k in list(st.session_state):
            if k.startswith("results_"):st.session_state.pop(k,None)
        st.rerun()
    with_flow=st.checkbox("국내 외국인·기관 순매수 포함(조회가 느릴 수 있음)",value=True)
    st.caption("수급 조회가 실패해도 기술점수(80점 만점)를 100점으로 환산해 표시합니다. 점수 기준 열에서 기술환산과 종합을 구분합니다.")
    st.caption("국내 수급 데이터 출처: K-Export Stars의 KIS 기반 파생 집계, CC BY 4.0 (기준일·조회 결과 확인 후 반영).")
    st.caption("서버 데이터 캐시: 시세 6시간, 종목 구성 24시간. 원격 무료 서버 첫 접속은 느릴 수 있습니다.")
    st.warning("자동매매 없음 · 점수는 사전 검증되지 않은 참고 규칙입니다.")

MARKET_LABELS={"KR":"🇰🇷 코스피 100 + TIGER 2개","KQ":"🇰🇷 코스닥 100","US":"🇺🇸 나스닥100 + SOXL"}
active=st.segmented_control("분석할 시장 선택",options=list(MARKET_LABELS),
    default="KR",format_func=lambda key:MARKET_LABELS[key],
    selection_mode="single",key="active_market")
if active is None:active="KR"
st.caption("현재 선택: "+MARKET_LABELS[active]+" — 결과표·상세 차트·시장별 수급 조회 대상에 함께 적용")
entries=listing[active]
st.caption("국내 구성 출처: "+listing["KR_source"])
st.caption("코스닥 구성 출처: "+listing["KQ_source"])
st.caption("미국 구성 출처: "+listing["US_source"])
if listing["KR_error"]:
    st.warning("코스피 시총 상위 100종목 목록을 조회하지 못했습니다. 필수 ETF 2개만 표시합니다. 임의 종목을 100개로 채우지 않습니다. 오류: "+listing["KR_error"][:230])

if listing["KQ_error"]:
    st.warning("코스닥 시총 상위 100종목 목록을 조회하지 못했습니다. 임의의 종목으로 채우지 않습니다. 오류: "+listing["KQ_error"][:230])

summaryA,summaryB,summaryC,summaryD=st.columns(4)
summaryA.metric("코스피 상위 기업 수",listing["KR_index_count"])
summaryB.metric("코스닥 상위 기업 수",listing["KQ_index_count"])
summaryC.metric("나스닥100 기업 수",listing["US_index_count"])
summaryD.metric("필수 별도 상품",3)

ranking_tab,candidate_tab,detail_tab,flow_tab,guide_tab=st.tabs(["📋 선택 시장 분석","🔎 투자 후보 탐색","📊 상세 차트","🏦 국내 수급","ℹ️ 분석 기준"])

def ranking_page(mkt):
    items=listing[mkt]
    if not items:
        st.error("현재 종목 목록을 조회할 수 없습니다.")
        return
    st.subheader("코스피 시총 100 + TIGER ETF 2종" if mkt=="KR" else "코스닥 시총 상위 100" if mkt=="KQ" else "Nasdaq-100 기업 100 + SOXL")
    st.caption("코스피 TIGER 2종 / 미국 SOXL 고정, 코스닥 100은 시총순. 나머지도 시장별 시총순 조회(나스닥 데이터 실패 시 예외). 50종목씩 분석합니다.")
    mandatories={"472150","0262B0"} if mkt=="KR" else {"SOXL"} if mkt=="US" else set()
    # The universe is pre-ranked by exchange market cap. Keep this stable order.
    ordered=[t for t in items if t[0] in mandatories]+[t for t in items if t[0] not in mandatories]
    order_index={code:i for i,(code,_,_,_) in enumerate(ordered)}
    market_rank={code:i for i,(code,_,_,_) in enumerate(
        (t for t in ordered if t[0] not in mandatories),start=1)}
    if mkt=="US" and not listing.get("US_cap_sorted",False):
        st.warning("나스닥 시가총액 자료 조회가 실패하여 현재 표시 순서는 정확한 시총순이 아닙니다. 시총 확인 시에만 시총순으로 표시합니다.")
    st.caption(f"현재 목록 {len(ordered)}개: "+", ".join(t[1] for t in ordered if t[0] in mandatories))
    if mkt in ("KR","KQ"):
        st.caption("구분의 코스피/코스닥 N위는 해당 시장 일반주 시총 순위입니다. ETF는 순위에서 제외합니다.")
        unknown=sum(t[3]=="업종 확인 불가" for t in ordered)
        if unknown:
            st.warning(f"업종 조회 실패 {unknown}개: 국내 구성 출처·조회 오류를 확인하세요.")
    key="results_v26_"+mkt+("_flow" if with_flow else "_tech")
    if key not in st.session_state:
        st.session_state[key]={"rows":{},"errors":{},"batches":[]}
    state=st.session_state[key]
    page_key="result_batch_page_v26_"+mkt+("_flow" if with_flow else "_tech")
    left,right=st.columns([3,1])
    remaining=[t for t in ordered if t[0] not in state["rows"] and t[0] not in state["errors"]]
    if left.button("▶ 다음 50종목 실제 시세 분석",key="run"+mkt,type="primary",
                   disabled=not remaining):
        batch=remaining[:50]
        # Record each attempted ticker by batch even if a data provider fails.
        # This prevents failures from shifting the following 50 onto the first page.
        state["batches"].append([item[0] for item in batch])
        st.session_state[page_key]=len(state["batches"])
        progress=st.progress(0,text=f"새 분석 0/{len(batch)}")
        for idx,item in enumerate(batch,1):
            try:
                state["rows"][item[0]]=to_row(item,with_flow=with_flow,market_rank=market_rank.get(item[0]))
            except Exception as e:
                state["errors"][item[0]]={"종목":item[1],"코드":item[0],"오류":str(e)[:260]}
            progress.progress(idx/len(batch),text=f"새 분석 {idx}/{len(batch)} — {item[1]}")
        progress.empty()
    if right.button("이 시장 결과 초기화",key="reset"+mkt):
        st.session_state[key]={"rows":{},"errors":{},"batches":[]}
        st.session_state.pop(page_key,None)
        st.rerun()
    if state["errors"] and st.button(f"↻ 조회 실패 {len(state['errors'])}개만 재시도",key="retry_"+mkt):
        lookup={t[0]:t for t in ordered}
        failed_codes=list(state["errors"])
        progress=st.progress(0,text="오류 종목 재조회")
        for idx,code in enumerate(failed_codes,1):
            if code in lookup:
                try:
                    state["rows"][code]=to_row(lookup[code],with_flow=with_flow,market_rank=market_rank.get(code))
                    state["errors"].pop(code,None)
                except Exception as e:
                    state["errors"][code]["오류"]=str(e)[:260]
            progress.progress(idx/len(failed_codes),text=f"오류 재시도 {idx}/{len(failed_codes)}")
        progress.empty()
    success=len(state["rows"]); fail=len(state["errors"])
    st.progress((success+fail)/len(ordered),text=f"분석 진행: {success+fail}/{len(ordered)} (정상 {success} · 오류 {fail})")
    if success:
        checked=list(state["rows"].values())
        disagreement=sum(x.get("종가 교차검증")=="종가 차이 큼·매수 보류" for x in checked)
        not_verified=sum(x.get("종가 교차검증")=="교차검증 불가" for x in checked)
        if disagreement or not_verified:
            st.caption(f"보조 시세 검토: 큰 차이(2% 초과) {disagreement}개 · 교차검증 불가 {not_verified}개. "
                       "작은 공급처 차이보다 상승여력·손절폭·손익비를 우선해서 비교하세요.")
        approved=sum(x.get("매매 판단") in ("매수 검토","강한 매수 검토") for x in checked)
        technical_only=sum(x.get("매매 판단") in ("연구용 매수 후보·실거래 미승인","연구용 후보·위험조건 미충족") for x in checked)
        gated=sum("매수 유보" in str(x.get("매매 판단","")) and
                  str(x.get("기술 판단(목표 검증 전)","")).startswith(("매수 검토","강한 매수 검토","기술매수 후보"))
                  for x in checked)
        st.caption(f"분석 완료 기준 수급 확인 매수 검토 {approved}개 · 기술매수 후보(수급 미확인) {technical_only}개 "
                   f"· 목표가·손익비로 유보 {gated}개 · 기타 {success-approved-technical_only-gated}개")
        if approved==0:
            st.info("매수 검토 0개는 매수할 주식이 없다는 뜻이 아닙니다. "
                    "수급 미확인 시에는 기술적 매수 후보로 분류하고, 과거 10거래일 경로·손절 위험으로 추가 유보될 수 있습니다. "
                    "기술 판단(목표 검증 전) 열과 매매 판단 근거를 확인하세요. 백테스트 검증 전 참고 신호입니다.")
    if success:
        df=pd.DataFrame(list(state["rows"].values()))
        df["_market_order"]=df["코드"].map(order_index)
        df=df.sort_values("_market_order",ascending=True).drop(columns="_market_order")
        # Display the exact batch that was just analyzed, not the first 50
        # successful rows. The next-analysis button automatically selects it.
        pages=len(state["batches"])
        if pages==0:
            # Graceful fallback for an old/stale Streamlit session.
            state["batches"]=[list(state["rows"])+list(state["errors"])]
            pages=1
        if st.session_state.get(page_key) not in range(1,pages+1):
            st.session_state[page_key]=pages
        page=st.selectbox("분석 결과 묶음 (최대 50개씩)",
                          list(range(1,pages+1)),key=page_key,
                          format_func=lambda p:f"{p}번째 50종목 / 총 {pages}묶음")
        # Prioritize actionable prices and the trading decision in the horizontal table.
        primary_cols=("구분","종목","코드","업종","거래일","통화",
                      "매매 판단","종가","통합 목표가","손절 참고","중장기 상승추세",
                      "목표 상승여력 %","손절폭 %","기대손익비(R)","과거 비용 반영 손익 %",
                      "2차 익절 참고","52주 고점","52주 저점")
        first=[col for col in primary_cols if col in df.columns]
        df=df[first+[col for col in df.columns if col not in first]]
        batch_codes=set(state["batches"][page-1])
        shown=df[df["코드"].isin(batch_codes)]
        batch_failed=sum(code in state["errors"] for code in batch_codes)
        st.caption(f"{page}번째 분석 묶음: 정상 {len(shown)}개 · 조회 실패 {batch_failed}개 "
                   f"| 누적 정상 {len(df)}개 · 누적 분석 {success+fail}/{len(ordered)}개")
        # Display thousands separators for every price/target/MA field,
        # without converting stored numeric values or changing CSV data.
        price_cols=("종가","20일선","60일선","120일선","200일선",
                    "52주 고점","52주 저점","손절 참고","통합 목표가","2차 익절 참고","과거 저항 참고",
                    "1차 저항 목표","2차 저항 목표","2R 손익비 목표")
        number_formats={col:("{:,.2f}" if mkt=="US" else "{:,.0f}") for col in price_cols if col in shown.columns}
        for col in ("점수","기술점수","기술만점","목표 분석점수"):
            if col in shown.columns:
                number_formats[col]="{:.0f}"
        # Net-buy figures are share counts: group thousands, no fractional shares.
        for col in ("외국인 5일 순매수(주)","외국인 20일 순매수(주)",
                    "기관 5일 순매수(주)","기관 20일 순매수(주)"):
            if col in shown.columns:
                number_formats[col]="{:,.0f}"
        # Keep meaningful indicator precision, hide insignificant trailing zeros.
        for col in ("RSI","RSI 5일 변화","ADX"):
            if col in shown.columns:
                number_formats[col]=lambda v: f"{v:.1f}".rstrip("0").rstrip(".")
        for col in ("거래량/20일 평균","1차 기대손익비(R)",
                    "2차 기대손익비(R)","통합목표 손익비(R)","기대손익비(R)"):
            if col in shown.columns:
                number_formats[col]=lambda v: f"{v:.2f}".rstrip("0").rstrip(".")
        if "전일대비 %" in shown.columns:
            number_formats["전일대비 %"]="{:.2f}"
        if "목표 상승여력 %" in shown.columns:
            number_formats["목표 상승여력 %"]="{:.2f}"
        if "손절폭 %" in shown.columns:
            number_formats["손절폭 %"]="{:.2f}"
        if "과거 비용 반영 손익 %" in shown.columns:
            number_formats["과거 비용 반영 손익 %"]="{:+.2f}"
        for col in ("과거 1차 익절 우선 %","과거 2차 익절 우선 %",
                    "과거 손절 우선 %"):
            if col in shown.columns:number_formats[col]="{:.1f}"
        if "과거 경로 표본" in shown.columns:number_formats["과거 경로 표본"]="{:,.0f}"
        if "종가 차이 %" in shown.columns:
            number_formats["종가 차이 %"]="{:.2f}"
        styled=shown.style.format(number_formats,na_rep="—")
        st.dataframe(styled,hide_index=True,use_container_width=True,height=510)
        st.download_button("📥 분석 결과 CSV 저장",df.to_csv(index=False).encode("utf-8-sig"),f"STOCK_RADAR200_{mkt}.csv","text/csv",key="csv"+mkt)
    else:
        st.info("위 버튼을 눌러 종목별 시세 분석을 시작하세요. 항목을 추측해서 채우지 않습니다.")
    if fail:
        with st.expander(f"⚠️ 조회 실패 {fail}개 — 원인 확인"):
            st.dataframe(pd.DataFrame(list(state["errors"].values())),hide_index=True,use_container_width=True)
    with st.expander(f"전체 구성 종목 {len(ordered)}개 보기 (50개씩)"):
        listing_pages=max(1,(len(ordered)+49)//50)
        listing_page=st.selectbox("종목 목록 페이지",list(range(1,listing_pages+1)),key="listing_page_"+mkt,
                                   format_func=lambda p:f"{p}페이지 / {listing_pages}페이지")
        page_items=ordered[(listing_page-1)*50:listing_page*50]
        st.dataframe(pd.DataFrame([{"구분":(f"코스피 {market_rank[c]}위" if mkt=="KR" else f"코스닥 {market_rank[c]}위" if mkt=="KQ" else f"나스닥 {market_rank[c]}위") if c in market_rank else ("국내 ETF" if mkt=="KR" else "미국 ETF"),"코드":c,"종목":n,"업종":s} for c,n,_,s in page_items]),use_container_width=True,hide_index=True,height=360)

with ranking_tab:ranking_page(active)

with candidate_tab:
    st.subheader("투자 후보 탐색 — "+MARKET_LABELS[active])
    st.caption("실제 분석된 종목만 표시합니다. 기술 상승 신호·수급·종가 검증을 별도로 구분해 비교합니다.")
    result_key="results_v26_"+active+("_flow" if with_flow else "_tech")
    result_state=st.session_state.get(result_key,{"rows":{}})
    if not result_state.get("rows"):
        st.info("선택 시장 분석 탭에서 먼저 50종목 실제 분석을 실행하세요.")
    else:
        candidates=pd.DataFrame(list(result_state["rows"].values()))
        st.caption(f'현재 분석 자료 {len(candidates)}개 — 전체 {len(listing[active])}개 종목 중 조회된 결과')
        groups={
            "전체":"전체",
            "연구용 후보(경고 포함)":"연구용 후보(경고 포함)",
            "기술적 상승 신호":"기술적 상승 신호",
            "종가 확인 필요":"종가 확인 필요",
            "매수 유보·관찰":"매수 유보·관찰"}
        filter_choice=st.selectbox("후보 필터",list(groups),key="candidate_filter_"+active)
        if filter_choice=="연구용 후보(경고 포함)":
            candidates=candidates[candidates["매매 판단"].isin(("연구용 매수 후보·실거래 미승인","연구용 후보·위험조건 미충족"))]
        elif filter_choice=="기술적 상승 신호":
            candidates=candidates[candidates["기술 판단(목표 검증 전)"].isin(
                ("매수 검토","강한 매수 검토","기술매수 후보·수급 미확인"))]
        elif filter_choice=="종가 확인 필요":
            candidates=candidates[candidates["종가 교차검증"]!="보조 시세 일치(공식값 미검증)"]
        elif filter_choice=="매수 유보·관찰":
            candidates=candidates[~candidates["매매 판단"].isin(("연구용 매수 후보·실거래 미승인","연구용 후보·위험조건 미충족"))]
        sort_col=st.selectbox("정렬 기준",["점수","목표 상승여력 %","기대손익비(R)"],
                              key="candidate_sort_"+active)
        candidates=candidates.sort_values(sort_col,ascending=False,na_position="last")
        columns=["구분","종목","코드","매매 판단","기술 판단(목표 검증 전)",
                 "시장 위험국면","실전 전략 승인","종가","통합 목표가","손절 참고","목표 상승여력 %","손절폭 %",
                 "기대손익비(R)","과거 비용 반영 손익 %",
                 "과거 1차 익절 우선 %","과거 경로 표본",
                 "점수","종가 교차검증","매매 판단 근거","거래일"]
        columns=[col for col in columns if col in candidates]
        st.caption(f"조건에 해당하는 종목 {len(candidates)}개")
        if len(candidates):
            fmt_cols={col:("{:,.2f}" if active=="US" and col!="점수" else "{:,.0f}") for col in ("종가","통합 목표가","손절 참고","점수") if col in columns}
            fmt_cols.update({col:"{:.2f}" for col in ("목표 상승여력 %","손절폭 %","기대손익비(R)","과거 1차 익절 우선 %","과거 비용 반영 손익 %") if col in columns})
            st.dataframe(candidates[columns].style.format(fmt_cols,na_rep="—"),
                         hide_index=True,use_container_width=True,height=520)
            st.download_button("📥 조건별 후보 CSV",candidates.to_csv(index=False).encode("utf-8-sig"),
                               f"STOCK_RADAR_candidates_{active}.csv","text/csv",key="candidate_csv_"+active)
        else:
            st.info("선택 조건에 해당하는 후보가 없습니다. 기술 신호나 종가 검증 상태를 확인하세요.")

with detail_tab:
    st.subheader("종목별 기술적 분석")
    current=listing[active]
    options={f'{name} ({code})':(code,name,m) for code,name,m,_ in current}
    if options:
        selected=st.selectbox("종목 검색 · 선택",list(options),key="detail_select_"+active)
        code,name,mkt=options[selected]
        if st.button("📈 차트 및 매매 판단 조회",key="getchart",type="primary"):
            st.session_state["detail_ticker_"+active]=selected
        if st.session_state.get("detail_ticker_"+active)==selected:
            try:
                data,src,r,flow_data,flow_error=analyze_item((code,name,mkt,""),with_flow=with_flow)
                st.caption(f'가격 기준일: {r["date"]} | 공급처: {src} | 통화: {"KRW" if mkt in ("KR","KQ") else "USD"}')
                a,b,c,d,e=st.columns(5)
                a.metric("종가",fmt(r["price"],mkt),f'{r["change"]:+.2f}%')
                b.metric("분석 점수",r["score"] if r["score"] is not None else "시세 부족")
                st.caption("점수 유형: "+r["score_basis"])
                c.metric("RSI14",fmt(r["rsi"]),f'{r["rsi_delta5"]:+.1f}p/5일' if r["rsi_delta5"] is not None else None)
                d.metric("ADX14",fmt(r["adx"]))
                e.metric("52주 고점",fmt(r["hi52"],mkt))
                qc=r["quote_check"]
                if qc["status"]=="mismatch":
                    st.warning(f'보조 공급처 종가 차이 {qc["difference_pct"]:.2f}% — 2% 초과 시 매수 판단 유보, 소폭 차이는 참고 표시')
                elif qc["status"]=="unverified":
                    st.warning("종가 독립 검증 불가: "+qc["reason"])
                else:
                    st.caption(f'같은 거래일 보조 종가 대조 통과(차이 {qc["difference_pct"]:.2f}%). 공식 확정 종가 검증은 아닙니다.')
                st.subheader("종합 판단: "+r["signal"])
                st.write(r["reason"] or "강한 매수 조건 미충족")
                if r["technical_signal_before_gate"]!=r["signal"]:
                    st.caption("목표가·손익비 적용 전 기술 판단: "+r["technical_signal_before_gate"])
                if r["signal"] in ("상승 신호·매수 유보","기술상승·매수 유보(수급 미확인)"):
                    st.warning("매수 유보 사유: "+r["entry_reason"])
                else:
                    st.caption("매수 목표 검증: "+r["entry_reason"])
                if r["upside_pct"] is not None:
                    st.caption(f'통합 목표가까지 상승 여력: {r["upside_pct"]:.2f}%'+
                               (f' | 손절가 대비 기대손익비: {r["entry_rr"]:.2f}R' if r["entry_rr"] is not None else ""))
                if r["score"] is None:
                    st.warning("200일 시세·거래량 등 기술분석 기초 자료가 부족하여 점수 계산을 유보합니다.")
                elif mkt in ("KR","KQ") and r["flow_status"]!="정상":
                    st.info("수급 조회 실패: 이 점수는 외국인·기관 점수를 제외한 기술분석 환산점수입니다. 종합점수와 직접 비교하지 마세요.")
                st.caption(f'기술점수: {r["technical_score"] if r["technical_score"] is not None else "—"}/{r["technical_max"]}점 | 국내 수급 상태: {r["flow_status"]}')
                if mkt in ("KR","KQ"):
                    if flow_data:
                        fstats=flow_data["외국인"]; istats=flow_data["기관"]
                        st.dataframe(pd.DataFrame([
                            {"투자자":"외국인","5거래일 순매수(주)":fstats["5일"],"20거래일 순매수(주)":fstats["20일"]},
                            {"투자자":"기관","5거래일 순매수(주)":istats["5일"],"20거래일 순매수(주)":istats["20일"]}
                        ]).style.format({"5거래일 순매수(주)":"{:,.0f}","20거래일 순매수(주)":"{:,.0f}"}),hide_index=True,use_container_width=True)
                    elif flow_error:st.caption("수급 조회 불가: "+flow_error)
                metrics=st.columns(len(r["scores"]))
                for col,(label,points) in zip(metrics,r["scores"].items()):
                    col.metric(label,points if points is not None and (label!="거래량·OBV" or r["volume_available"]) else "자료 없음")
                st.caption("국내: 추세 25 + 모멘텀 25 + ADX 15 + 거래량·OBV 15 + 외국인·기관 20 | 미국: 추세 25 + 모멘텀 25 + ADX 20 + 거래량·OBV 30")
                v=data.tail(252)
                fig=make_subplots(rows=4,cols=1,shared_xaxes=True,row_heights=[.53,.15,.16,.16],vertical_spacing=.025)
                fig.add_trace(go.Candlestick(x=v.index,open=v.Open,high=v.High,low=v.Low,close=v.Close,name="주가"),row=1,col=1)
                for n,color in [(20,"#e9ae54"),(60,"#4d9cdf"),(120,"#21c7af"),(200,"#b683df")]:
                    fig.add_trace(go.Scatter(x=v.index,y=v[f"MA{n}"],name=f"{n}일선",line=dict(color=color,width=1.6)),row=1,col=1)
                fig.add_trace(go.Bar(x=v.index,y=v.Volume,name="거래량"),row=2,col=1)
                fig.add_trace(go.Scatter(x=v.index,y=v.RSI,name="RSI"),row=3,col=1)
                for y in (30,70):fig.add_hline(y=y,row=3,col=1,line_dash="dash")
                for column,label,color in [("ADX","ADX","#30c5b8"),("PDI","+DI","#6fcb6f"),("MDI","-DI","#df7373")]:
                    fig.add_trace(go.Scatter(x=v.index,y=v[column],name=label,line=dict(color=color)),row=4,col=1)
                fig.add_hline(y=25,row=4,col=1,line_dash="dash")
                for key,label,color in (("stop","손절 참고","#d94848"),("trade_target","1차 익절","#14885c"),
                                        ("trade_target2","2차 익절","#3476a1")):
                    if r[key] is not None:
                        fig.add_hline(y=r[key],row=1,col=1,line_dash="dot",line_color=color,
                                      annotation_text=label,annotation_position="top left")
                fig.update_layout(height=840,xaxis_rangeslider_visible=False,margin=dict(l=8,r=8,t=18,b=8))
                st.plotly_chart(fig,use_container_width=True)
                st.subheader("익절·손절 매매 계획 (과거 10거래일 경로 기준)")
                p1,p2,p3,p4=st.columns(4)
                p1.metric("1차 익절 참고",fmt(r["trade_target"],mkt))
                p2.metric("2차 익절 참고",fmt(r["trade_target2"],mkt))
                p3.metric("손절폭",f'{r["stop_risk_pct"]:.2f}%' if r["stop_risk_pct"] is not None else "—")
                p4.metric("익절·손절 손익비",f'{r["entry_rr"]:.2f}R' if r["entry_rr"] is not None else "—")
                st.caption("익절 산정: "+r["trade_plan_basis"])
                if r["historical_net_expectancy_pct"] is not None:
                    st.caption(f'과거 동일 규칙 손절·익절 후 평균 순손익: '
                               f'{r["historical_net_expectancy_pct"]:+.2f}% '
                               f'(왕복비용 0.35% 가정, 향후 수익률 예측 아님)')
                if r["historical_target_first_pct"] is not None:
                    st.caption(f'과거 경로에서 1차 익절이 손절보다 먼저 관측된 비율 '
                               f'{r["historical_target_first_pct"]:.1f}% · 2차 {r["historical_second_first_pct"]:.1f}%'
                               f' · 손절 먼저 {r["historical_stop_first_pct"]:.1f}%'
                               ' (그 외는 기간 내 미도달; 향후 달성 확률 아님)')
                with st.expander("💰 실제 투입금액 대비 손절·익절 예상 금액",expanded=True):
                    unit="원" if mkt in ("KR","KQ") else "달러"
                    default_capital=10000000.0 if unit=="원" else 10000.0
                    capital=st.number_input("투입 가능한 최대 자금 ("+unit+")",min_value=1.0,
                        value=default_capital,step=100000.0 if unit=="원" else 100.0,
                        key="risk_capital_"+mkt)
                    risk_percent=st.slider("한 거래에서 감수할 최대 손절 위험 (% 자금)",
                        min_value=0.5,max_value=5.0,value=1.0,step=0.5,key="risk_percent_"+mkt)
                    trading_cost=st.number_input("왕복 비용·슬리피지 가정 (%)",
                        min_value=0.0,max_value=5.0,value=0.35,step=0.05,
                        key="risk_cost_"+mkt)
                    sizing=position_sizing(capital,r["price"],r["stop"],r["trade_target"],
                        risk_fraction=risk_percent/100,
                        round_trip_cost_fraction=trading_cost/100)
                    s1,s2,s3,s4=st.columns(4)
                    s1.metric("참고 매수수량",f'{sizing["shares"]:,}주')
                    s2.metric("예상 투자원금",f'{sizing["cost"]:,.2f} {unit}' if unit=="달러"
                              else f'{sizing["cost"]:,.0f} {unit}')
                    s3.metric("손절 시 계획손실",f'-{sizing["planned_loss"]:,.0f} {unit}'
                              if sizing["planned_loss"] is not None else "산정 불가")
                    s4.metric("1차 익절 시 계획수익",f'+{sizing["planned_profit"]:,.0f} {unit}'
                              if sizing["planned_profit"] is not None else "산정 불가")
                    st.caption("종가 진입·손절 참고가 체결 가정입니다. 왕복 비용·슬리피지를 가정치로 반영하며 "
                               "세금·갭 하락·실제 체결·시장충격은 별개입니다. 손실이 계획보다 커질 수 있습니다.")
                st.caption("과거 저항 참고: "+fmt(r["integrated_target"],mkt)+" / 종합 근거: "+(r["target_evidence"] or "분석 가능한 자료 없음"))
                st.caption("과거 매물대 분석 방식: "+(r["volume_profile"] or "자료 부족")+
                           " | 외국인·기관: "+r["target_inputs"])
                m1,m2,m3,m4=st.columns(4)
                m1.metric("손절 참고",fmt(r["stop"],mkt))
                m2.metric("1차 저항 목표",fmt(r["target1"],mkt))
                m3.metric("2차 저항 목표",fmt(r["target2"],mkt))
                m4.metric("2R 손익비 참고",fmt(r["target"],mkt))
                st.caption("1차 저항 근거: "+(r["target1_basis"] or "현재가 위의 확인된 저항 없음")+
                           " | 2차 저항 근거: "+(r["target2_basis"] or "추가 저항 미확인"))
                st.caption("익절 참고가는 과거에 관찰된 10거래일 유리변동의 중앙값·75백분위이고, 저항 목표와는 별개입니다. "
                           "고가 도달이 실제 매매 수익이나 목표 달성 확률을 보장하지 않습니다. 손절 주문도 갭 하락 시 지정 금액으로 체결되지 않을 수 있습니다.")
                if r["target1_r"] is not None:
                    st.caption(f'1차 저항까지 손익비: {r["target1_r"]:.2f}R'+
                               (f' | 2차 저항까지: {r["target2_r"]:.2f}R' if r["target2_r"] is not None else ""))
                if code=="SOXL":
                    st.warning("SOXL은 일일 3배 레버리지 ETF입니다. 변동성이 크므로 과거 저항이 쉽게 무너질 수 있고 장기 보유 수익률은 기초지수의 단순 3배가 아닙니다.")
                if code in ("472150","0262B0"):
                    st.info("커버드콜 ETF의 저항 목표는 주가 기준입니다. 분배금을 포함한 총수익 목표가 아니며 분배락·분배정책을 별도로 고려해야 합니다.")
            except Exception as e:st.error(f"실제 시세 조회 실패: {e}")
    else:st.error("이 시장의 목록 데이터가 없습니다.")

with flow_tab:
    st.subheader("국내 기관·외국인 순매수")
    kr_entries=listing[active] if active in ("KR","KQ") else []
    if active=="US":
        st.info("미국 시장은 국내식 외국인·기관 순매수 집계를 적용하지 않습니다. 코스피 또는 코스닥을 선택하세요.")
    elif kr_entries:
        sel=st.selectbox("국내 종목",range(len(kr_entries)),key="flow_select_"+active,
                         format_func=lambda i:f"{kr_entries[i][1]} ({kr_entries[i][0]})")
        code=kr_entries[sel][0]
        if st.button("🏦 순매수 조회",key="flows_"+active):
            try:
                data,src=priced(code,active)
                result=flows(code,str(data.index[-1].date()))
                st.caption("KRX 수급 기준일 "+result["date"])
                flow_df=pd.DataFrame([{"구분":investor,**stats} for investor,stats in result.items() if investor in ("기관","외국인")])
                flow_cols={col:"{:,.0f}" for col in ("1일","5일","20일","연속 순매수일") if col in flow_df.columns}
                st.dataframe(flow_df.style.format(flow_cols,na_rep="—"),hide_index=True,use_container_width=True)
            except Exception as e:st.error(f"공개 수급 자료 조회 불가: {e}")
    elif active!="US":st.warning("이 국내 시장의 종목 구성을 조회하지 못했습니다.")

with guide_tab:
    st.markdown("""**종목 선정**
- 코스피: FinanceDataReader의 코스피 시가총액 상위 100 일반 종목을 조회합니다. *공식 코스피100 지수 편입종목과 동일하다고 주장하지 않습니다.*
- 코스닥: FinanceDataReader의 코스닥 시가총액 상위 100 일반 종목을 별도로 조회합니다. ETF·스팩·우선주를 제외하며 종목수 부족·시총 조회 실패 시 임의 목록으로 대체하지 않습니다.
- 나스닥: Nasdaq-100 지수에 편입된 약 100개 기업을 기준으로, 알파벳 복수 주식 종류는 GOOGL만 표시합니다. 공개 구성 목록 조회 실패 시 2026-11-13에 저장한 100종목 목록을 사용하고 출처를 명시합니다.
- 예외 상품: TIGER 배당커버드콜액티브(472150), TIGER 200커버드콜액티브(0262B0), SOXL을 항상 따로 포함합니다.

**종합 점수**
- **국내 100점:** 20·60·120·200일선 추세 25점, RSI/MACD 25점, ADX/DI 15점, 거래량·OBV 15점, 외국인·기관 5·20거래일 누적 순매수 20점
- **미국 100점:** 20·60·120·200일선 추세 25점, RSI/MACD 25점, ADX/DI 20점, 거래량·OBV 30점. 국내식 외국인·기관 수급은 적용하지 않습니다.
- **국내 수급 가점:** 외국인 5일·20일 누적, 기관 5일·20일 누적 순매수가 각각 양수이면 5점씩(총 20점). 음수와 실제 0은 0점, 자료 누락은 평가 불가
- **국내 수급 출처:** [K-Export Stars 외국인·기관 파생 집계](https://github.com/james-brand/korea-market-data), CC BY 4.0. 한국투자증권 API 기반 가공자료이며 원자료와 별개입니다. 공개 기준일이 7일 넘게 오래되면 자동으로 사용을 거절합니다. ETF는 집계 대상에서 제외될 수 있습니다.
- 수급 누락 시 **기술점수(80점 만점)를 100점으로 환산**해 표시하되, 점수 기준에 **기술환산(수급 제외)**이라고 구분합니다. 종합점수와 직접 비교하지 마세요.
- 변동성 보조: 볼린저밴드, ATR. 200일 가격·거래량 자료 부족 시 기술점수도 유보합니다.

**통합 목표가 산정 (규칙 기반, 제한된 과거 표본 검증만 시행)**
- **저항 가격 후보:** 직전 20·60·120·252일 고점, 3거래일 좌우 국소 고점, 현재가 위의 20·60·120일 이동평균선, 거래량 집중 가격대(일별 대표가격×거래량 히스토그램 근사). 가까운 가격대는 저항 구간 하나로 합칩니다.
- **거래량·돌파:** 현재가가 직전 20일 고점을 넘어가고 거래량이 20일 평균의 1.5배 이상인지 확인합니다.
- **상승 여건:** 20·60·120일 정배열, RSI 범위와 MACD 히스토그램 방향, 가까운 거래량 집중 저항 유무, 국내 주식은 외국인·기관의 5·20일 누적 순매수를 함께 평가합니다. 수급 미조회 시 **수급 제외**라고 명시합니다.
- **참고 저항 선택:** 과거 고점과 매물대는 가격 장애물로 따로 표시합니다. 이 저항값을 주요 익절가로 사용하지 않습니다. 익절 참고는 과거 완료된 10거래일 경로의 유리변동 중앙값·75백분위로 산정하며, 미래 수익이나 도달 확률 예측이 아닙니다.
- **매물대 제한:** 거래량 집중 가격대는 일봉 대표가격에 하루 거래량을 배정한 **근사치**입니다. 실제 가격별 체결량·매물대 분석과 동일하지 않습니다.
- **2R 손익비 목표:** ATR과 20일선을 참고해 정한 손절폭의 두 배를 현재가에 더한 값입니다. 통합 목표가와 별개입니다.
- **신규 매수 검증:** 기술적 매수 신호여도 과거 완료된 10거래일 가격 경로에서 관측한 유리변동 중앙값 2% 이상, 1차 익절/손절 손익비 1.2R 이상, 손절폭 12% 이하여야 매수 검토를 유지합니다. 자료가 부족하면 유보합니다. 수급 미확인은 별도 후보로 표시하고 과열·하락 경고를 우선합니다. 이 기준은 **백테스트로 우수성이 입증되지 않은 시험 운영값**입니다.
- **수급 조회 실패:** 기술점수가 60점 이상이면 **기술매수 후보·수급 미확인**으로 별도 표시하고, 목표·손익비 미달은 **기술상승·매수 유보(수급 미확인)**으로 표기합니다. 이를 외국인·기관 매수가 확인된 실제 **매수 검토**와 혼동하지 마세요. 수급 조회 설정을 바꾸면 이전 분석 결과와 별개로 저장됩니다.
- **ETF 주의:** SOXL의 3배 레버리지와 TIGER 커버드콜 ETF의 분배금·분배락은 저항 목표 산식에 반영되지 않습니다.

**시세 품질:** 동일 거래일 종가를 Yahoo 보조 시세와 대조하고 0.5% 초과 차이는 참고 경고로 표시합니다. **2% 초과 차이만 신규 매수 신호를 유보**합니다. 무료 시세는 공식 KRX 확정 종가와 대조하지 않았으며 OHLCV 전체의 정확성을 보증하지 않습니다. 실제 주문 전 증권사 종가·거래일을 확인하세요.

**주의사항:** 과거 7개 선정 종목 표본에서 검증한 결과로는 이 점수의 수익률 우위를 입증하지 못했습니다. 과거 수급과 거래비용이 없는 제한적 진단이며, 전략 수익률을 보장하지 않습니다. 무료 호스팅과 외부 데이터 공급자의 사정으로 일부 종목 조회가 실패할 수 있으며, 실시간 시세/정시 자동 갱신/자동매매 기능은 없습니다.""")

    audit100_path=Path("data/trade-strategy-audit.json")
    if audit100_path.exists():
        try:
            audit100=json.loads(audit100_path.read_text(encoding="utf8"))
            intended=audit100.get("requested_tickers",7)
            verified=audit100.get("analysed_tickers")
            with st.expander("📊 코스피·코스닥·나스닥 각 100종목 (총 300종목) 백테스트"):
                if intended!=300:
                    st.info("300종목 검증 결과는 아직 저장되지 않았습니다. 아래 정보는 최신 완료된 이전 표본 자료입니다.")
                else:
                    st.caption(f'검증 기준일(UTC) {audit100.get("time_utc","—")} '
                               f'| 요청 {intended}종목 · 분석 {verified}종목 · 실패 {audit100.get("failed_tickers")}종목 '
                               f'| 매매 신호 {audit100.get("total_trades",0)}건')
                    gate=audit100.get("release_gate",{})
                    if not gate:
                        st.warning("실전 판매용 독립 검증은 아직 완료되지 않았습니다. 과거 구간 재검증만으로 합격을 선언하지 않습니다.")
                    if gate:
                        if gate.get("status")=="EXPLORATORY_PASS_NOT_INDEPENDENT":
                            st.warning("과거 구간을 다시 사용한 탐색적 수익 기준은 충족했지만, 이미 살펴본 기간이므로 "
                                       "독립 검증 합격이나 실제 판매 승인으로 인정하지 않습니다.")
                        elif gate.get("status")=="PASS" and gate.get("independent_validation_passed"):
                            st.success("독립 검증 수익 기준 통과. 규제·데이터 사용권과 실제 운영 검사는 별도입니다.")
                        else:
                            st.error("투자전략 성과 검증: 미통과 — 실제 운용/유료 판매를 위한 기준에 미달합니다. "
                                     f"검증 구간 매매 건수 {gate.get('holdout_trades','—')}건 "
                                     f"(최소 {gate.get('minimum_holdout_trades',50)}건).")
                    aggregate=audit100.get("aggregate",{})
                    c1,c2,c3=st.columns(3)
                    c1.metric("총 매매 거래",f'{audit100.get("total_trades",0):,}건')
                    c2.metric("매매 승률",f'{aggregate["win_rate_pct"]:.2f}%' if aggregate.get("win_rate_pct") is not None else "측정 불가")
                    c3.metric("평균 거래 순수익률",f'{aggregate["mean_trade_net_pct"]:+.2f}%' if aggregate.get("mean_trade_net_pct") is not None else "측정 불가")
                    st.caption(f'시점별 전체 판정 기록 {audit100.get("historical_decision_count","—")}회 · 왕복비용 {audit100.get("round_trip_cost_pct",0.35)}% 가정')
                    walk=audit100.get("variant_holdout_audit",{})
                    if walk.get("results"):
                        st.markdown("**손절·익절 전략 비교 (과거/검증 구간을 날짜로 분리)**")
                        st.caption(f'분리 기준일: {walk.get("split_date")} | 이전 구간에서 검토하고 이후 구간은 별도로 공개')
                        versions={"TAKE_STOP":"현행 익절·손절",
                                  "HOLD_10":"10거래일 보유",
                                  "STOP_10":"고정 손절 + 10일 보유",
                                  "HALF_TRAIL":"절반 익절 + 나머지 추적손절"}
                        table=[]
                        for variant,label in versions.items():
                            dev=walk["results"].get("development",{}).get(variant,{})
                            hold=walk["results"].get("holdout",{}).get(variant,{})
                            table.append({"전략":label,"과거 건수":dev.get("n"),
                                          "과거 평균 순수익 %":dev.get("mean_net_return_pct"),
                                          "검증 건수":hold.get("n"),
                                          "검증 평균 순수익 %":hold.get("mean_net_return_pct"),
                                          "검증 승률 %":hold.get("win_rate_pct")})
                        st.dataframe(pd.DataFrame(table),hide_index=True,use_container_width=True)
                        st.caption("검증 구간이 단순 보유보다 좋지 않다면 매매전략을 합격 처리하지 않습니다. "
                                   "종목 구성 생존편향과 같은 종목의 반복 거래 등 한계는 그대로 남습니다.")
                    directions=audit100.get("signal_direction_audit",{})
                    if directions:
                        st.markdown("**매매 판단별 실제 10거래일 후 방향 비교**")
                        st.dataframe(pd.DataFrame([{"판정":signal,**values} for signal,values in directions.items()]),
                                     hide_index=True,use_container_width=True)
                    for market,stats in audit100.get("per_market",{}).items():
                        st.caption(f'{market}: 종목 {stats.get("analysed")}/{stats.get("requested")} · 거래 {stats.get("trades")}')
                        if stats.get("signal_direction_audit"):
                            st.dataframe(pd.DataFrame([{"판정":signal,**values}
                                                      for signal,values in stats["signal_direction_audit"].items()]),
                                         hide_index=True,use_container_width=True)
                    outcomes=[{"시장·종목":k,"종목명":v.get("name",""),"거래 수":v.get("trades",0),
                               "평균 순수익률 %":v.get("mean_trade_net_pct"),
                               "승률 %":v.get("win_rate_pct"),
                               "오류":v.get("error","")}
                              for k,v in audit100.get("tickers",{}).items()]
                    if outcomes:st.dataframe(pd.DataFrame(outcomes),hide_index=True,use_container_width=True)
                    st.warning("현 시총순 상위 종목을 과거로 되돌린 표본이므로 생존편향이 있습니다. "
                               "과거 수급 부재·동시 보유·시장충격·지수 대비 초과성과 검증이 빠져 있습니다. "
                               "유료 판매의 정확성·성과 인증 결과로 사용할 수 없습니다.")
        except Exception as e:
            st.warning("300종목 백테스트 결과 조회 오류: "+str(e)[:120])
    with st.expander("🧪 시세 교차확인·과거신호 검증 결과 (제한된 표본)"):
        audit_path=Path("data/accuracy-audit.json")
        if audit_path.exists():
            audit=json.loads(audit_path.read_text(encoding="utf8"))
            st.caption("진단 수행 시각(UTC): "+audit.get("as_of_utc","미상")+
                       " — 표본 7개 종목, 공식 거래소 종가 검증 및 시장 전체 백테스트 아님")
            quote_rows=[{"시장":q["market"],"종목코드":q["ticker"],
                         "보조 종가 검증":q["comparison"].get("status"),
                         "시세 차이 %":q["comparison"].get("relative_difference_pct")}
                        for q in audit.get("quote_comparisons",[])]
            if quote_rows:
                st.dataframe(pd.DataFrame(quote_rows),hide_index=True,use_container_width=True)
            back=audit.get("backtest",{})
            pooled=back.get("pooled_selected_sample",{})
            st.caption(f'과거 10거래일 후 수익률의 전체 표본: {back.get("total_observations","—")}건 '
                       "— 표본 선정 편향·관측기간 중복·거래비용·과거 수급 미반영")
            if pooled:
                st.dataframe(pd.DataFrame([{"신호":signal,**values} for signal,values in pooled.items()]),
                             hide_index=True,use_container_width=True)
            st.warning("이 제한적 진단에서 통합 목표가·매수판단의 우수성은 입증되지 않았습니다. 실제 주문 전 공식 확정 시세를 따로 대조해야 합니다.")
        else:
            st.info("자동 진단 파일이 생성되지 않아 검증 결과를 표시할 수 없습니다.")
