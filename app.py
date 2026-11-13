import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from universe import get_universe
from engine import compute,evaluate
from providers import get_prices,get_flows

st.set_page_config(page_title="STOCK RADAR 200",page_icon="📈",layout="wide")
st.markdown("<style>.block-container{padding-top:1.4rem;max-width:1560px}.stMetric{padding:12px;border:1px solid #64748b44;border-radius:12px}</style>",unsafe_allow_html=True)
st.title("📈 STOCK RADAR 200")
st.caption("🇰🇷 코스피 시총 상위 100 + TIGER ETF 2개 | 🇺🇸 Nasdaq-100 기업 100 + SOXL | 지표별 종가 기준 분석")

@st.cache_data(ttl=86400,show_spinner=False)
def universe():return get_universe()

@st.cache_data(ttl=21600,show_spinner=False)
def priced(code,market):
    d,source=get_prices(code,market)
    return compute(d),source

@st.cache_data(ttl=10800,show_spinner=False)
def flows(code,date):return get_flows(code,date)

try: listing=universe()
except Exception as e:
    st.error(f"종목 목록 초기화 실패: {e}")
    st.stop()

def analyze_item(item,with_flow=True):
    code,name,market,sector=item
    data,src=priced(code,market)
    flow_data=None
    flow_error=None
    if market=="KR" and with_flow:
        try:
            flow_data=flows(code,str(data.index[-1].date()))
        except Exception as exc:
            flow_error=str(exc)[:160]
    r=evaluate(data,market=market,flow=flow_data)
    return data,src,r,flow_data,flow_error

def to_row(item,with_flow=True,market_rank=None):
    code,name,market,sector=item
    data,src,r,flow_data,flow_error=analyze_item(item,with_flow=with_flow)
    f=flow_data or {}
    foreign=f.get("외국인",{})
    institution=f.get("기관",{})
    return {
      "구분":(f"코스피 {market_rank}위" if market_rank is not None else "국내 ETF") if market=="KR" else (f"나스닥 {market_rank}위" if market_rank is not None else "미국 ETF"),
      "종목":name,"코드":code,"업종":sector,
      "거래일":r["date"],"통화":"KRW" if market=="KR" else "USD",
      "종가":round(r["price"],2),"전일대비 %":round(r["change"],2),
      "점수":r["score"],"점수 기준":r["score_basis"],"기술점수":r["technical_score"],"기술만점":r["technical_max"],
      "매매 판단":r["signal"],
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
      "통합 목표가":round(r["integrated_target"],2) if r["integrated_target"] is not None else None,
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
      "시세 출처":src}

def fmt(v):return "—" if v is None else f"{v:,.0f}"

with st.sidebar:
    st.header("조회 설정")
    st.info("코스피 100종목과 나스닥 100기업을 별도로 분석합니다. 필수 ETF 3개는 항상 추가됩니다.")
    market=st.radio("분석 시장",["🇰🇷 코스피 100 + TIGER 2개","🇺🇸 나스닥100 + SOXL"])
    active="KR" if "코스피" in market else "US"
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

entries=listing[active]
st.caption("국내 구성 출처: "+listing["KR_source"])
st.caption("미국 구성 출처: "+listing["US_source"])
if listing["KR_error"]:
    st.warning("코스피 시총 상위 100종목 목록을 조회하지 못했습니다. 필수 ETF 2개만 표시합니다. 임의 종목을 100개로 채우지 않습니다. 오류: "+listing["KR_error"][:230])

summaryA,summaryB,summaryC=st.columns(3)
summaryA.metric("코스피 상위 기업 수",listing["KR_index_count"])
summaryB.metric("나스닥100 기업 수",listing["US_index_count"])
summaryC.metric("필수 별도 상품",3)

kr_tab,us_tab,detail_tab,flow_tab,guide_tab=st.tabs(["🇰🇷 코스피 100","🇺🇸 나스닥 100","📊 상세 차트","🏦 국내 수급","ℹ️ 분석 기준"])

def ranking_page(mkt):
    items=listing[mkt]
    if not items:
        st.error("현재 종목 목록을 조회할 수 없습니다.")
        return
    st.subheader(("코스피 시총 100 + TIGER ETF 2종" if mkt=="KR" else "Nasdaq-100 기업 100 + SOXL"))
    st.caption("국내 TIGER 2종 / 미국 SOXL 상단 고정, 이후 시가총액 내림차순. 분석 점수는 순위 정렬에 사용하지 않습니다. 50종목씩 조회합니다.")
    mandatories={"472150","0262B0"} if mkt=="KR" else {"SOXL"}
    # The universe is pre-ranked by exchange market cap. Keep this stable order.
    ordered=[t for t in items if t[0] in mandatories]+[t for t in items if t[0] not in mandatories]
    order_index={code:i for i,(code,_,_,_) in enumerate(ordered)}
    market_rank={code:i for i,(code,_,_,_) in enumerate(
        (t for t in ordered if t[0] not in mandatories),start=1)}
    if mkt=="US" and not listing.get("US_cap_sorted",False):
        st.warning("나스닥 시가총액 자료 조회가 실패하여 현재 표시 순서는 정확한 시총순이 아닙니다. 시총 확인 시에만 시총순으로 표시합니다.")
    st.caption(f"현재 목록 {len(ordered)}개: "+", ".join(t[1] for t in ordered if t[0] in mandatories))
    if mkt=="KR":
        st.caption("구분의 코스피 N위는 조회일 기준 일반주 시가총액 순위입니다. ETF는 순위에서 제외합니다.")
        unknown=sum(t[3]=="업종 확인 불가" for t in ordered)
        if unknown:
            st.warning(f"업종 조회 실패 {unknown}개: 국내 구성 출처·조회 오류를 확인하세요.")
    key="results_v12_"+mkt
    if key not in st.session_state:st.session_state[key]={"rows":{},"errors":{}}
    state=st.session_state[key]
    left,right=st.columns([3,1])
    if left.button("▶ 다음 50종목 실제 시세 분석",key="run"+mkt,type="primary"):
        remaining=[t for t in ordered if t[0] not in state["rows"] and t[0] not in state["errors"]]
        with st.spinner(f"{min(50,len(remaining))}개 종목의 실제 OHLCV 조회 중"):
            for item in remaining[:50]:
                try:state["rows"][item[0]]=to_row(item,with_flow=with_flow,market_rank=market_rank.get(item[0]))
                except Exception as e:state["errors"][item[0]]={"종목":item[1],"코드":item[0],"오류":str(e)[:260]}
    if right.button("이 시장 결과 초기화",key="reset"+mkt):
        st.session_state[key]={"rows":{},"errors":{}}
        st.rerun()
    success=len(state["rows"]); fail=len(state["errors"])
    st.progress((success+fail)/len(ordered),text=f"분석 진행: {success+fail}/{len(ordered)} (정상 {success} · 오류 {fail})")
    if success:
        df=pd.DataFrame(list(state["rows"].values()))
        df["_market_order"]=df["코드"].map(order_index)
        df=df.sort_values("_market_order",ascending=True).drop(columns="_market_order")
        # Display at most 50 analyzed rows at once; preserve full CSV download.
        pages=max(1,(len(df)+49)//50)
        page=st.selectbox("분석 결과 페이지 (50개씩)",list(range(1,pages+1)),key="result_page_"+mkt,
                          format_func=lambda p:f"{p}페이지 / {pages}페이지")
        # Prioritize actionable prices and the trading decision in the horizontal table.
        primary_cols=("구분","종목","코드","종가","손절 참고","통합 목표가",
                      "52주 고점","52주 저점","매매 판단")
        first=[col for col in primary_cols if col in df.columns]
        df=df[first+[col for col in df.columns if col not in first]]
        shown=df.iloc[(page-1)*50:page*50]
        st.caption(f"분석 완료 {len(df)}종목 중 {len(shown)}종목 표시")
        # Display thousands separators for every price/target/MA field,
        # without converting stored numeric values or changing CSV data.
        price_cols=("종가","20일선","60일선","120일선","200일선",
                    "52주 고점","52주 저점","손절 참고","통합 목표가",
                    "1차 저항 목표","2차 저항 목표","2R 손익비 목표")
        number_formats={col:"{:,.0f}" for col in price_cols if col in shown.columns}
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
                    "2차 기대손익비(R)","통합목표 손익비(R)"):
            if col in shown.columns:
                number_formats[col]=lambda v: f"{v:.2f}".rstrip("0").rstrip(".")
        if "전일대비 %" in shown.columns:
            number_formats["전일대비 %"]="{:.2f}"
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
        st.dataframe(pd.DataFrame([{"구분":(f"코스피 {market_rank[c]}위" if mkt=="KR" else f"나스닥 {market_rank[c]}위") if c in market_rank else ("국내 ETF" if mkt=="KR" else "미국 ETF"),"코드":c,"종목":n,"업종":s} for c,n,_,s in page_items]),use_container_width=True,hide_index=True,height=360)

with kr_tab:ranking_page("KR")
with us_tab:ranking_page("US")

with detail_tab:
    st.subheader("종목별 기술적 분석")
    current=listing[active]
    options={f'{name} ({code})':(code,name,m) for code,name,m,_ in current}
    if options:
        selected=st.selectbox("종목 검색 · 선택",list(options),key="detail_select")
        code,name,mkt=options[selected]
        if st.button("📈 차트 및 매매 판단 조회",key="getchart",type="primary"):
            st.session_state["detail_ticker"]=selected
        if st.session_state.get("detail_ticker")==selected:
            try:
                data,src,r,flow_data,flow_error=analyze_item((code,name,mkt,""),with_flow=with_flow)
                st.caption(f'가격 기준일: {r["date"]} | 공급처: {src} | 통화: {"KRW" if mkt=="KR" else "USD"}')
                a,b,c,d,e=st.columns(5)
                a.metric("종가",fmt(r["price"]),f'{r["change"]:+.2f}%')
                b.metric("분석 점수",r["score"] if r["score"] is not None else "시세 부족")
                st.caption("점수 유형: "+r["score_basis"])
                c.metric("RSI14",fmt(r["rsi"]),f'{r["rsi_delta5"]:+.1f}p/5일' if r["rsi_delta5"] is not None else None)
                d.metric("ADX14",fmt(r["adx"]))
                e.metric("52주 고점",fmt(r["hi52"]))
                st.subheader("종합 판단: "+r["signal"])
                st.write(r["reason"] or "강한 매수 조건 미충족")
                if r["score"] is None:
                    st.warning("200일 시세·거래량 등 기술분석 기초 자료가 부족하여 점수 계산을 유보합니다.")
                elif mkt=="KR" and r["flow_status"]!="정상":
                    st.info("수급 조회 실패: 이 점수는 외국인·기관 점수를 제외한 기술분석 환산점수입니다. 종합점수와 직접 비교하지 마세요.")
                st.caption(f'기술점수: {r["technical_score"] if r["technical_score"] is not None else "—"}/{r["technical_max"]}점 | 국내 수급 상태: {r["flow_status"]}')
                if mkt=="KR":
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
                fig.update_layout(height=840,xaxis_rangeslider_visible=False,margin=dict(l=8,r=8,t=18,b=8))
                st.plotly_chart(fig,use_container_width=True)
                st.subheader("통합 기술·수급 목표가")
                p1,p2,p3,p4=st.columns(4)
                p1.metric("통합 목표가",fmt(r["integrated_target"]))
                p2.metric("목표 달성 여건",r["target_grade"])
                p3.metric("목표 근거 점수",f'{r["target_signal_points"]}/100' if r["target_signal_points"] is not None else "자료 부족")
                p4.metric("통합목표 손익비",f'{r["target_r"]:.2f}R' if r["target_r"] is not None else "—")
                st.caption("종합 근거: "+(r["target_evidence"] or "분석 가능한 자료 없음"))
                st.caption("과거 매물대 분석 방식: "+(r["volume_profile"] or "자료 부족")+
                           " | 외국인·기관: "+r["target_inputs"])
                m1,m2,m3,m4=st.columns(4)
                m1.metric("손절 참고",fmt(r["stop"]))
                m2.metric("1차 저항 목표",fmt(r["target1"]))
                m3.metric("2차 저항 목표",fmt(r["target2"]))
                m4.metric("2R 손익비 참고",fmt(r["target"]))
                st.caption("1차 저항 근거: "+(r["target1_basis"] or "현재가 위의 확인된 저항 없음")+
                           " | 2차 저항 근거: "+(r["target2_basis"] or "추가 저항 미확인"))
                st.caption("통합 목표가는 확인된 1·2차 저항대에서 거래량 돌파·이평선·RSI·MACD·수급의 상승 여건을 반영해 선택합니다. 2R은 별개이며 목표가 도달 확률을 뜻하지 않습니다.")
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
    kr_entries=listing["KR"]
    if kr_entries:
        sel=st.selectbox("국내 종목",range(len(kr_entries)),format_func=lambda i:f"{kr_entries[i][1]} ({kr_entries[i][0]})")
        code=kr_entries[sel][0]
        if st.button("🏦 순매수 조회",key="flows"):
            try:
                data,src=priced(code,"KR")
                result=flows(code,str(data.index[-1].date()))
                st.caption("KRX 수급 기준일 "+result["date"])
                flow_df=pd.DataFrame([{"구분":investor,**stats} for investor,stats in result.items() if investor in ("기관","외국인")])
                flow_cols={col:"{:,.0f}" for col in ("1일","5일","20일","연속 순매수일") if col in flow_df.columns}
                st.dataframe(flow_df.style.format(flow_cols,na_rep="—"),hide_index=True,use_container_width=True)
            except Exception as e:st.error(f"공개 수급 자료 조회 불가: {e}")
    else:st.warning("코스피 구성 조회 실패. 다음에 새로고침해 주세요.")

with guide_tab:
    st.markdown("""**종목 선정**
- 코스피: FinanceDataReader의 코스피 시가총액 상위 100 일반 종목을 조회합니다. *공식 코스피100 지수 편입종목과 동일하다고 주장하지 않습니다.*
- 나스닥: Nasdaq-100 지수에 편입된 약 100개 기업을 기준으로, 알파벳 복수 주식 종류는 GOOGL만 표시합니다. 공개 구성 목록 조회 실패 시 2026-11-13에 저장한 100종목 목록을 사용하고 출처를 명시합니다.
- 예외 상품: TIGER 배당커버드콜액티브(472150), TIGER 200커버드콜액티브(0262B0), SOXL을 항상 따로 포함합니다.

**종합 점수**
- **국내 100점:** 20·60·120·200일선 추세 25점, RSI/MACD 25점, ADX/DI 15점, 거래량·OBV 15점, 외국인·기관 5·20거래일 누적 순매수 20점
- **미국 100점:** 20·60·120·200일선 추세 25점, RSI/MACD 25점, ADX/DI 20점, 거래량·OBV 30점. 국내식 외국인·기관 수급은 적용하지 않습니다.
- **국내 수급 가점:** 외국인 5일·20일 누적, 기관 5일·20일 누적 순매수가 각각 양수이면 5점씩(총 20점). 음수와 실제 0은 0점, 자료 누락은 평가 불가
- **국내 수급 출처:** [K-Export Stars 외국인·기관 파생 집계](https://github.com/james-brand/korea-market-data), CC BY 4.0. 한국투자증권 API 기반 가공자료이며 원자료와 별개입니다. 공개 기준일이 7일 넘게 오래되면 자동으로 사용을 거절합니다. ETF는 집계 대상에서 제외될 수 있습니다.
- 수급 누락 시 **기술점수(80점 만점)를 100점으로 환산**해 표시하되, 점수 기준에 **기술환산(수급 제외)**이라고 구분합니다. 종합점수와 직접 비교하지 마세요.
- 변동성 보조: 볼린저밴드, ATR. 200일 가격·거래량 자료 부족 시 기술점수도 유보합니다.

**통합 목표가 산정 (규칙 기반, 백테스트 미실시)**
- **저항 가격 후보:** 직전 20·60·120·252일 고점, 3거래일 좌우 국소 고점, 현재가 위의 20·60·120일 이동평균선, 거래량 집중 가격대(일별 대표가격×거래량 히스토그램 근사). 가까운 가격대는 저항 구간 하나로 합칩니다.
- **거래량·돌파:** 현재가가 직전 20일 고점을 넘어가고 거래량이 20일 평균의 1.5배 이상인지 확인합니다.
- **상승 여건:** 20·60·120일 정배열, RSI 범위와 MACD 히스토그램 방향, 가까운 거래량 집중 저항 유무, 국내 주식은 외국인·기관의 5·20일 누적 순매수를 함께 평가합니다. 수급 미조회 시 **수급 제외**라고 명시합니다.
- **통합 목표 선택:** 위 조건이 강하고 거래량 돌파까지 확인되면 2차 저항을 선택할 수 있으며, 그렇지 않으면 가까운 1차 저항을 선택합니다. 기준은 **개발자가 정한 휴리스틱**으로, 실제 목표 달성 확률·적정주가를 의미하지 않습니다.
- **매물대 제한:** 거래량 집중 가격대는 일봉 대표가격에 하루 거래량을 배정한 **근사치**입니다. 실제 가격별 체결량·매물대 분석과 동일하지 않습니다.
- **2R 손익비 목표:** ATR과 20일선을 참고해 정한 손절폭의 두 배를 현재가에 더한 값입니다. 통합 목표가와 별개입니다.
- **ETF 주의:** SOXL의 3배 레버리지와 TIGER 커버드콜 ETF의 분배금·분배락은 저항 목표 산식에 반영되지 않습니다.

**주의사항:** 이 점수는 백테스트로 수익률이 입증되지 않은 규칙 기반 분석입니다. 무료 호스팅과 외부 데이터 공급자의 사정으로 일부 종목 조회가 실패할 수 있으며, 실시간 시세/정시 자동 갱신/자동매매 기능은 없습니다.""")
