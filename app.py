import datetime as dt
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from universe import WATCHLIST, KR
from engine import compute, evaluate
from providers import get_prices, get_flows

st.set_page_config(page_title="STOCK RADAR 50",page_icon="📈",layout="wide")
st.markdown("<style>.block-container{padding-top:1.3rem;max-width:1560px}.stMetric{padding:12px;border:1px solid #64748b44;border-radius:12px}</style>",unsafe_allow_html=True)
st.title("📈 STOCK RADAR 50")
st.caption("🇰🇷 국내 39종목 우선 | 🇺🇸 미국 11종목 | 서버에서 시세 수집 · 종가 기준 매매 참고 분석")

@st.cache_data(ttl=21600,show_spinner=False)
def priced(code,market):
    df,source=get_prices(code,market)
    return compute(df),source

@st.cache_data(ttl=10800,show_spinner=False)
def flows(code,date):return get_flows(code,date)

@st.cache_data(ttl=21600,show_spinner=False)
def scan():
    rows=[];errors=[]
    for c,n,m,s in WATCHLIST:
        try:
            d,src=priced(c,m)
            r=evaluate(d)
            rows.append({"시장":"국내" if m=="KR" else "미국","종목":n,"코드":c,"분야":s,
                "거래일":r["date"],"통화":"KRW" if m=="KR" else "USD",
                "종가":round(r["price"],2),"등락률 %":round(r["change"],2),
                "점수":r["score"],"판정":r["signal"],
                "RSI":round(r["rsi"],1) if r["rsi"] is not None else None,
                "RSI 5일 변화":round(r["rsi_delta5"],1) if r["rsi_delta5"] is not None else None,
                "RSI 방향":r["rsi_direction"],
                "RSI 거래량 확인":"확인" if r["rsi_volume_confirmed"] else "미확인",
                "52주 고점":round(r["hi52"],2),"52주 저점":round(r["lo52"],2),
                "거래량 배율":round(r["volratio"],2) if r["volratio"] else None,
                "손절 참고":round(r["stop"],2) if r["stop"] else None,
                "목표 참고":round(r["target"],2) if r["target"] else None,"출처":src})
        except Exception as e:
            errors.append({"코드":c,"종목":n,"시장":m,"오류":str(e)[:260]})
    return pd.DataFrame(rows),pd.DataFrame(errors)

def f(v):
    return "—" if v is None else f"{v:,.2f}"

with st.sidebar:
    st.header("설정")
    st.info("주가는 웹서버에서 수집합니다. 사무실 PC에는 프로그램을 설치할 필요가 없습니다.")
    mfilter=st.radio("시장",["국내 우선 전체","국내만","미국만"])
    if st.button("🔄 데이터 다시 가져오기",use_container_width=True):
        st.cache_data.clear()
        st.session_state.pop("scan",None)
        st.rerun()
    st.caption("캐시 주기 6시간 · 무료 호스팅은 사용하지 않을 때 잠들 수 있습니다.")
    st.warning("자동 주문 없음. 규칙 기반 참고 신호이며 투자 수익을 보장하지 않습니다.")

rank,chart,flow,verification,guide=st.tabs(["🏆 50종목 순위","📊 상세 차트","🏦 기관·외국인 수급","✅ 50종목 시세 검증","ℹ️ 안내"])

with rank:
    st.subheader("50종목 종합 모니터")
    st.caption("국내 주식부터 순차 조회합니다. 오류가 발생한 종목은 별도로 표시하며 시세를 지어내지 않습니다.")
    if st.button("▶ 50종목 분석 시작",type="primary"):
        with st.spinner("외부 서버가 실제 시세를 조회 중입니다"):
            st.session_state["scan"]=scan()
    if "scan" not in st.session_state:
        st.info("분석 시작을 누르거나 차트 탭에서 개별 종목부터 조회해 보세요.")
    else:
        data,err=st.session_state["scan"]
        a,b,c,d=st.columns(4)
        a.metric("정상 분석",f"{len(data)}/50")
        b.metric("매수 검토 60+",int((data["점수"]>=60).sum()) if len(data) else 0)
        c.metric("장기자료 부족",int(data["점수"].isna().sum()) if len(data) else 0)
        d.metric("조회 오류",len(err))
        if len(data):
            view=data.copy()
            if mfilter=="국내만":view=view[view["시장"]=="국내"]
            elif mfilter=="미국만":view=view[view["시장"]=="미국"]
            view["order"]=view["시장"].map({"국내":0,"미국":1})
            view=view.sort_values(["order","점수"],ascending=[True,False],na_position="last").drop(columns="order")
            st.dataframe(view,hide_index=True,use_container_width=True,height=560)
            st.download_button("📥 CSV 다운로드",view.to_csv(index=False).encode("utf-8-sig"),"STOCK_RADAR50.csv","text/csv")
        if len(err):
            with st.expander(f"⚠️ 조회 실패 {len(err)}종목"):
                st.dataframe(err,hide_index=True,use_container_width=True)

with chart:
    options={f'{"🇰🇷" if m=="KR" else "🇺🇸"} {n} ({c})':(c,n,m) for c,n,m,_ in WATCHLIST}
    sel=st.selectbox("종목 선택",list(options))
    code,name,market=options[sel]
    if st.button("📈 차트 및 신호 조회",type="primary",key="details"):st.session_state["detail"]=sel
    if st.session_state.get("detail")==sel:
        try:
            data,src=priced(code,market)
            r=evaluate(data)
            st.caption(f'기준 거래일: {r["date"]} | 공급원: {src} | 통화: {"KRW" if market=="KR" else "USD"}')
            x1,x2,x3,x4,x5=st.columns(5)
            x1.metric("최근 종가",f(r["price"]),f'{r["change"]:+.2f}%')
            x2.metric("점수",str(r["score"]) if r["score"] is not None else "데이터 부족")
            x3.metric("RSI14",f(r["rsi"]),f'{r["rsi_delta5"]:+.1f}p / 5거래일' if r["rsi_delta5"] is not None else None)
            st.caption(f'RSI 최근 5거래일: {r["rsi_direction"]} | RSI 상승·거래량 동시 확인: {"예" if r["rsi_volume_confirmed"] else "아니오"}')
            x4.metric("52주 고점",f(r["hi52"]))
            x5.metric("52주 저점",f(r["lo52"]))
            st.subheader(f'판정: {r["signal"]}')
            st.write(r["reason"] or "강한 상승 조건이 부족합니다")
            if r["score"] is None:st.warning("장기 데이터가 200거래일 미만이므로 점수를 계산하지 않습니다.")
            view=data.tail(252)
            fig=make_subplots(rows=3,cols=1,shared_xaxes=True,row_heights=[.67,.17,.16],vertical_spacing=.03)
            fig.add_trace(go.Candlestick(x=view.index,open=view.Open,high=view.High,low=view.Low,close=view.Close,name="주가"),row=1,col=1)
            for n,color in [(20,"#e9ae54"),(60,"#4d9cdf"),(200,"#b683df")]:
                fig.add_trace(go.Scatter(x=view.index,y=view[f"MA{n}"],name=f"{n}일선",line=dict(color=color,width=1.7)),row=1,col=1)
            fig.add_trace(go.Bar(x=view.index,y=view.Volume,name="거래량"),row=2,col=1)
            fig.add_trace(go.Scatter(x=view.index,y=view.RSI,name="RSI"),row=3,col=1)
            fig.add_hline(y=30,row=3,col=1,line_dash="dash")
            fig.add_hline(y=70,row=3,col=1,line_dash="dash")
            fig.update_layout(height=700,xaxis_rangeslider_visible=False,margin=dict(l=8,r=8,t=20,b=8))
            st.plotly_chart(fig,use_container_width=True)
            a,b,c,d=st.columns(4)
            a.metric("20일 지지 참고",f(r["support"]))
            b.metric("20일 저항 참고",f(r["resistance"]))
            c.metric("손절 참고",f(r["stop"]))
            d.metric("2R 목표 참고",f(r["target"]))
            if code in ("SOXL","TQQQ"):st.error("3배 일일 레버리지 ETF: 큰 변동성과 손실에 주의하세요.")
        except Exception as e:st.error(f"시세 서버 조회 실패: {e}")

with flow:
    kchoices={f"{n} ({c})":c for c,n,_ in KR}
    selected=st.selectbox("국내 종목 선택",list(kchoices))
    code=kchoices[selected]
    if st.button("🏦 수급 가져오기",type="primary",key="flow"):
        try:
            data,src=priced(code,"KR")
            r=evaluate(data)
            result=flows(code,r["date"])
            st.caption(f'KRX 순매수 기준일: {result["date"]}')
            rows=[]
            for investor in ("기관","외국인"):
                v=result[investor]
                rows.append({"투자자":investor,"당일 순매수(주)":v["1일"],"5일 누적(주)":v["5일"],"20일 누적(주)":v["20일"],"연속 순매수일":v["연속 순매수일"]})
            st.dataframe(pd.DataFrame(rows),hide_index=True,use_container_width=True)
        except Exception as e:st.error(str(e))
    st.caption("수급값은 KRX 공개자료 조회가 가능한 경우에만 표시합니다. 조회 실패 시 0으로 표시하지 않습니다.")

with verification:
    st.subheader("50종목 실제 시세 수신 검증")
    st.caption("외부 검증 환경에서 종목별 가격과 거래일을 조회합니다. Render 서버의 연결 상태와는 다를 수 있습니다.")
    import json
    from pathlib import Path
    report_file=Path(__file__).resolve().parent/"data"/"verification.json"
    if not report_file.exists():
        st.warning("검증 결과가 아직 없습니다. 검증 전에는 모든 종목이 정상이라고 표시하지 않습니다.")
        st.markdown("[자동 검증 실행 상태 확인](https://github.com/beat3610-coder/stock-radar-50/actions)")
    else:
        report=json.loads(report_file.read_text(encoding="utf-8"))
        counters=report["counts"]
        st.caption(f'검증 시각(UTC): {report["checked_at_utc"]} | 검증 환경: {report["environment"]}')
        a,b,c=st.columns(3)
        a.metric("정상 조회",f'{counters.get("정상",0)}/50')
        b.metric("오래된 데이터",counters.get("기준일 오래됨",0))
        c.metric("조회 실패",counters.get("조회 실패",0))
        if counters.get("정상",0)<50:
            st.warning("전체 50종목 검증은 아직 통과하지 못했습니다. 실패 종목과 오류를 확인하세요.")
        else:
            st.success("외부 검사 환경에서 50종목 최신 시세가 정상 조회됐습니다. Render 서버 수집 성공은 별도로 확인해야 합니다.")
        audit=pd.DataFrame(report["rows"])
        st.dataframe(audit,hide_index=True,use_container_width=True,height=620)
        st.download_button("📥 시세 검증 결과 CSV",audit.to_csv(index=False).encode("utf-8-sig"),"STOCK_RADAR50_시세검증.csv","text/csv")
    st.info("상세 차트 탭을 통해 실제 Render 서버에서 개별 시세 조회 결과를 확인할 수 있습니다.")

with guide:
    st.markdown("""**STOCK RADAR 50의 판단 규칙**
- 20/60/200일 이동평균, MACD, RSI, 거래량, 20일 고점 돌파로 최대 100점 계산
- RSI 45~68 +5점, 5거래일 상승폭 3p 이상(현재 RSI 35~70) +5점
- 거래량 동반 상승(20일 평균의 1.5배 초과) +5점, RSI 상승과 거래량 동반 확인(1.2배 이상) +5점
- RSI가 5거래일 동안 5p 이상 하락하면 −5점, RSI가 75 초과면 −15점
- 200일선 하회 시 감점
- RSI의 상승·하락은 확정된 일봉에 기반한 규칙일 뿐, 반등·추세 전환을 보장하지 않음
- 장기 시세가 200거래일 미만인 신규 ETF는 점수를 유보
- 목표가와 손절가는 ATR 기반 모델 참고값이며 미래 가격 예측이 아님

**중요**: 이 웹앱은 무료 호스팅의 수집·캐시 제약이 있으며, 정해진 시각의 일일 자동 갱신이나 실시간 주가, 자동매매 기능을 보장하지 않습니다. 무료 데이터 제공자의 네트워크 정책에 따라 일부 시세와 수급은 조회되지 않을 수 있습니다. 데이터 최신 거래일을 확인하세요.""")
