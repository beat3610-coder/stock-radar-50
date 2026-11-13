import numpy as np
import pandas as pd

NEEDED = ["Open", "High", "Low", "Close", "Volume"]

def normalize(data):
    if data is None or data.empty: raise ValueError("시세 데이터 없음")
    x=data.copy()
    if isinstance(x.columns,pd.MultiIndex): x.columns=x.columns.get_level_values(0)
    x.columns=[str(c).title() for c in x.columns]
    if not all(c in x for c in NEEDED): raise ValueError("OHLCV 항목 누락")
    x=x[NEEDED].apply(pd.to_numeric,errors="coerce").dropna()
    x=x[(x.Close>0)&(x.Open>0)&(x.Low>0)&(x.High>=x.Low)&(x.Volume>=0)]
    x=x.loc[~x.index.duplicated(keep="last")].sort_index()
    # Raw (not dividend-adjusted) OHLC must be internally consistent.
    # Do not silently trade on malformed latest bars.
    invalid=~(x.Open.between(x.Low*0.999,x.High*1.001) &
              x.Close.between(x.Low*0.999,x.High*1.001))
    if len(x) and bool(invalid.iloc[-1]):
        raise ValueError("최신 OHLC 종가·시가가 고가/저가 범위를 벗어남 — 시세 공급처 확인 필요")
    if len(x) and float(invalid.mean())>0.02:
        raise ValueError("과거 OHLC 이상 비중 2% 초과 — 지표 계산 보류")
    x=x.loc[~invalid]
    if len(x)<2: raise ValueError(f"가격 계산에 필요한 최소 거래일 부족: {len(x)}일")
    return x

def compute(data):
    x=normalize(data)
    c=x.Close
    for n in (5,20,60,120,200): x[f"MA{n}"]=c.rolling(n).mean()
    d=c.diff()
    g=d.clip(lower=0).ewm(alpha=1/14,adjust=False,min_periods=14).mean()
    l=(-d.clip(upper=0)).ewm(alpha=1/14,adjust=False,min_periods=14).mean()
    rs=g/l.replace(0,np.nan)
    x["RSI"]=100-100/(1+rs)
    x.loc[(l==0)&(g>0),"RSI"]=100
    x.loc[(g==0)&(l>0),"RSI"]=0
    x.loc[(g==0)&(l==0),"RSI"]=50
    m=c.ewm(span=12,adjust=False,min_periods=12).mean()-c.ewm(span=26,adjust=False,min_periods=26).mean()
    x["MACD"]=m
    x["MACDS"]=m.ewm(span=9,adjust=False,min_periods=9).mean()
    x["MACDH"]=m-x.MACDS
    x["VOL20"]=x.Volume.rolling(20).mean()
    x["H52"]=x.High.rolling(252,min_periods=1).max()
    x["L52"]=x.Low.rolling(252,min_periods=1).min()
    x["RES"]=x.High.shift(1).rolling(20).max()
    x["SUP"]=x.Low.shift(1).rolling(20).min()
    tr=pd.concat([x.High-x.Low,(x.High-c.shift()).abs(),(x.Low-c.shift()).abs()],axis=1).max(axis=1)
    x["ATR"]=tr.ewm(alpha=1/14,adjust=False,min_periods=14).mean()
    # ADX(14): Wilder-style smoothed directional movement; avoid 0/0.
    up=x.High.diff(); down=-x.Low.diff()
    plus=up.where((up>down)&(up>0),0.0)
    minus=down.where((down>up)&(down>0),0.0)
    atr=x["ATR"].replace(0,np.nan)
    pdi=100*plus.ewm(alpha=1/14,adjust=False,min_periods=14).mean()/atr
    mdi=100*minus.ewm(alpha=1/14,adjust=False,min_periods=14).mean()/atr
    dx=100*(pdi-mdi).abs()/(pdi+mdi).replace(0,np.nan)
    x["ADX"]=dx.ewm(alpha=1/14,adjust=False,min_periods=14).mean()
    x["PDI"]=pdi
    x["MDI"]=mdi
    # OBV and its 20-day mean distinguish buying pressure from price alone.
    x["OBV"]=(np.sign(c.diff()).fillna(0)*x.Volume).cumsum()
    x["OBV20"]=x.OBV.rolling(20).mean()
    # Bollinger(20,2) is a risk context, not another momentum bonus.
    x["BBMID"]=c.rolling(20).mean()
    sd=c.rolling(20).std(ddof=0)
    x["BBUP"]=x.BBMID+2*sd
    x["BBLOW"]=x.BBMID-2*sd
    return x

def finite(v): return pd.notna(v) and bool(np.isfinite(v))


def integrated_targets(x, market, flow, stop):
    """Evidence-driven trade-plan target levels, not statistical price forecasts.

    Volume-by-price is a DAILY OHLCV proxy (typical price with full-day volume);
    actual traded-at-price volume requires intraday/tick data and is NOT claimed.
    Signal strength controls whether the nearer or farther resistance is a
    supportable scenario. No unsupported upside price is invented.
    """
    last=x.iloc[-1]
    p=float(last.Close)
    prior=x.iloc[:-1].tail(252)
    empty={"target1":None,"target2":None,"target1_basis":None,"target2_basis":None,
           "integrated_target":None,"target_grade":"산정 불가","target_evidence":"",
           "target1_r":None,"target2_r":None,"target_r":None,
           "volume_profile":None,"target_signal_points":None,
           "resistance_breakout":None,"target_inputs":"자료 부족"}
    if len(prior)<120 or not finite(last.ATR) or last.ATR<=0:
        return empty
    atr=float(last.ATR)
    gap=max(p*0.006,atr*0.25)
    levels=[]
    # Prior high-price levels, excluding the current incomplete candle.
    for n in (20,60,120,252):
        if len(prior)<n:continue
        val=float(prior.High.iloc[-n:].max())
        if val>p+gap:levels.append((val,f"{n}일 과거 고점","고점"))
    # Prior pivot resistance, strictly historical (no lookahead from today).
    h=prior.High.astype(float)
    if len(h)>=7:
        prev3=h.shift(1).rolling(3,min_periods=3).max()
        next3=h.iloc[::-1].shift(1).rolling(3,min_periods=3).max().iloc[::-1]
        pivots=h[(h>prev3)&(h>=next3)].tail(24)
        for price in pivots:
            if float(price)>p+gap:levels.append((float(price),"국소 고점","고점"))
    # Price/volume histogram: one daily typical-price sample per session.
    # It is an APPROXIMATION of cost-basis/supply zones, not an order-book.
    vol=prior.loc[prior.Volume.gt(0)].copy()
    profile_note="일별 대표가격×거래량 근사"
    if len(vol)>=60:
        typical=(vol.High+vol.Low+vol.Close)/3
        lower=float(min(vol.Low.min(),p))
        upper=float(max(vol.High.max(),p+gap))
        if upper>lower:
            bins=32
            weights=np.histogram(typical,bins=bins,range=(lower,upper),weights=vol.Volume)[0]
            edges=np.linspace(lower,upper,bins+1)
            top=float(max(weights)) if len(weights) else 0
            if top>0:
                for idx,w in enumerate(weights):
                    midpoint=float((edges[idx]+edges[idx+1])/2)
                    if midpoint>p+gap and w>=0.35*top and w>0:
                        levels.append((midpoint,"거래량 집중 가격대(근사)","매물대"))
    else:
        profile_note="거래량 자료 부족"
    # Moving averages can be overhead resistance when above current price.
    for n in (20,60,120):
        key=f"MA{n}"
        if finite(last[key]) and float(last[key])>p+gap:
            levels.append((float(last[key]),f"{n}일 이동평균선","이평선"))

    levels.sort(key=lambda z:z[0])
    merged=[]
    for price,label,kind in levels:
        if merged and price-merged[-1]["price"]<gap:
            # Combine close levels into one cluster, with evidence sources.
            merged[-1]["tags"].add(label)
            merged[-1]["kinds"].add(kind)
        else:
            merged.append({"price":price,"tags":{label},"kinds":{kind}})
    if not merged:
        result=empty.copy()
        result["target_evidence"]="현재가 위에서 확인된 저항 없음"
        result["volume_profile"]=profile_note
        return result

    first=merged[0]
    second=merged[1] if len(merged)>1 else None
    # Classify RSI/MACD, breakout+volume, MA trend and investor net buying.
    evidence=[]
    points=0;possible=0
    possible+=20
    vratio=float(last.Volume/last.VOL20) if finite(last.VOL20) and last.VOL20>0 else None
    prev20=float(prior.High.tail(20).max())
    breakout=bool(p>prev20 and vratio is not None and vratio>=1.5)
    if breakout:points+=20;evidence.append("20일 고점·거래량 동반 돌파")
    elif p>prev20:evidence.append("고점 돌파, 거래량 미확인")
    else:evidence.append("20일 고점 거래량 돌파 미확인")
    possible+=15
    ordered_ma=all(finite(last[f"MA{n}"]) for n in (20,60,120))
    if ordered_ma and p>last.MA20>last.MA60>last.MA120:
        points+=15;evidence.append("20·60·120일 정배열")
    elif ordered_ma and p>last.MA20:
        points+=7;evidence.append("20일선 상회")
    else:evidence.append("이평선 추세 미흡")
    possible+=20
    if finite(last.RSI):
        rsi=float(last.RSI)
        if 48<=rsi<=68:
            points+=20;evidence.append("RSI 상승 가능 구간")
        elif 40<=rsi<48 or 68<rsi<=72:
            points+=10;evidence.append("RSI 중립·과열 경계")
        else:evidence.append("RSI 약세/과열")
    else:evidence.append("RSI 정보 없음")
    possible+=20
    if finite(last.MACDH):
        if last.MACDH>0 and len(x)>1 and finite(x.MACDH.iloc[-2]) and last.MACDH>x.MACDH.iloc[-2]:
            points+=20;evidence.append("MACD 양수·상승")
        elif last.MACDH>0:
            points+=12;evidence.append("MACD 양수")
        else:evidence.append("MACD 상승 모멘텀 미흡")
    else:evidence.append("MACD 정보 없음")
    # Include volume-profile resistance as a hurdle: large overhead
    # supply is less supportive of an aggressive (second resistance) target.
    possible+=10
    near_overhead=bool("매물대" in first["kinds"] and first["price"]<p+max(atr,p*0.035))
    if near_overhead:
        evidence.append("가까운 거래량 집중 저항")
    else:
        points+=10;evidence.append("단기 집중 매물대 저항 제한적")
    flow_covered=False
    if market in ("KR","KQ"):
        if isinstance(flow,dict):
            try:
                f=flow["외국인"];ins=flow["기관"]
                gap_days=(pd.Timestamp(x.index[-1]).normalize()-pd.Timestamp(flow["date"]).normalize()).days
                vals=[float(f["5일"]),float(f["20일"]),float(ins["5일"]),float(ins["20일"])]
                if 0<=gap_days<=7 and all(np.isfinite(z) for z in vals):
                    flow_covered=True
                    possible+=15
                    points+=sum(z>0 for z in vals)*3.75
                    evidence.append(f"수급 외국인·기관 5·20일 순매수 {sum(z>0 for z in vals)}/4")
            except (KeyError,TypeError,ValueError):
                pass
        if not flow_covered:
            evidence.append("외국인·기관 수급 미확인(평가 제외)")
    else:
        evidence.append("미국 종목: 국내식 외국인·기관 수급 미적용")
    strength=int(round(100*points/possible))
    # The 'integrated target' is an OBSERVED resistance selected on evidence.
    # Avoid projecting beyond observed supportable supply levels.
    selected=second if second is not None and strength>=75 and breakout and not near_overhead else first
    grade=("상승 여건 강함" if strength>=75 else
           "상승 여건 보통" if strength>=50 else "상승 여건 약함")
    if market in ("KR","KQ") and not flow_covered:grade+=" · 수급 제외"
    def rr(level):
        return (level-p)/(p-stop) if level is not None and stop is not None and p>stop else None
    return {
        "target1":first["price"],
        "target2":second["price"] if second else None,
        "target1_basis":" · ".join(sorted(first["tags"])),
        "target2_basis":" · ".join(sorted(second["tags"])) if second else None,
        "integrated_target":selected["price"],
        "target_grade":grade,
        "target_evidence":" / ".join(evidence),
        "target1_r":rr(first["price"]),
        "target2_r":rr(second["price"]) if second else None,
        "target_r":rr(selected["price"]),
        "volume_profile":profile_note,
        "target_signal_points":strength,
        "resistance_breakout":breakout,
        "target_inputs":"수급 포함" if flow_covered else "수급 제외"
    }

def apply_target_risk_gate(signal, price, stop, target, min_upside_pct=3.0, min_rr=1.5):
    """Guard BUY recommendations with observable target upside and stop-based R/R.

    This is a configurable, unbacktested entry filter, not a return forecast.
    Never override non-buy risk warnings or recommend buying without inputs.
    """
    upside=(100.0*(target/price-1.0)) if (
        price is not None and price>0 and target is not None and np.isfinite(target)
    ) else None
    rr=(target-price)/(price-stop) if (
        upside is not None and stop is not None and np.isfinite(stop) and 0<stop<price
    ) else None
    buy_candidates=("강한 매수 검토","매수 검토","기술매수 후보·수급 미확인")
    if signal not in buy_candidates:
        return signal,upside,rr,"기존 기술적 위험 판단 유지"
    # Technical-only candidates are not full buy recommendations when flows are missing.
    hold_label=("기술상승·매수 유보(수급 미확인)" if signal=="기술매수 후보·수급 미확인"
                else "상승 신호·매수 유보")
    if upside is None or rr is None:
        return hold_label,upside,rr,"목표가 또는 손절가 불확실"
    reasons=[]
    if upside<min_upside_pct:
        reasons.append(f"목표 상승여력 {upside:.2f}% < 기준 {min_upside_pct:.1f}%")
    if rr<min_rr:
        reasons.append(f"목표 손익비 {rr:.2f}R < 기준 {min_rr:.1f}R")
    if reasons:
        return hold_label,upside,rr," · ".join(reasons)
    return signal,upside,rr,f"목표 상승여력 {upside:.2f}% · 기대손익비 {rr:.2f}R"


def evaluate(x, market="US", flow=None):
    """Rules-based daily signal. KR: 80 technical + 20 investor flow.
    US: 100 technical only. Missing KR flow never implies zero purchases.
    """
    r=x.iloc[-1]
    prev=x.iloc[-2]
    p=float(r.Close)
    def num(v):return float(v) if finite(v) else None
    def ok(k):return finite(r[k])
    rsi_prev=x.RSI.iloc[-6] if len(x)>=6 else np.nan
    rsi_delta=float(r.RSI-rsi_prev) if ok("RSI") and finite(rsi_prev) else None
    volume_available=bool(ok("VOL20") and r.VOL20>0 and ok("Volume") and r.Volume>0)
    volratio=float(r.Volume/r.VOL20) if volume_available else None
    domestic=market in ("KR","KQ")
    scores={"추세":0,"모멘텀":0,"추세 강도":0,"거래량·OBV":0}
    if domestic:scores["외국인·기관"]=None
    details=[]
    def award(group, cond, points, name):
        if cond:
            scores[group]+=points
            details.append(name)

    # Both markets: 20/60/120/200-day averages, total 25.
    award("추세",ok("MA20") and p>r.MA20,5,"20일선 위")
    award("추세",ok("MA60") and r.MA20>r.MA60,7,"20일선 > 60일선")
    award("추세",ok("MA120") and r.MA60>r.MA120,7,"60일선 > 120일선")
    award("추세",ok("MA200") and r.MA120>r.MA200 and p>r.MA200,6,"120일선 > 200일선")

    # Momentum: 25, same for each market.
    award("모멘텀",ok("MACDH") and r.MACDH>0,10,"MACD 양수")
    award("모멘텀",ok("RSI") and 45<=r.RSI<=68,8,"RSI 45~68")
    award("모멘텀",rsi_delta is not None and rsi_delta>=3 and r.RSI<=70,7,"RSI 5거래일 상승")

    # Strength: KR 15, US 20.
    strength_hi=15 if domestic else 20
    strength_mid=8 if domestic else 10
    award("추세 강도",ok("ADX") and r.ADX>=25 and ok("PDI") and ok("MDI") and r.PDI>r.MDI,
          strength_hi,"ADX 25 이상 상승 우세")
    award("추세 강도",ok("ADX") and 20<=r.ADX<25 and ok("PDI") and ok("MDI") and r.PDI>r.MDI,
          strength_mid,"ADX 20 이상 상승 우세")

    # Trading activity: KR 15, US 30. No data -> total score withheld.
    vol_high=5 if domestic else 12
    obv_high=7 if domestic else 12
    breakout=3 if domestic else 6
    award("거래량·OBV",volratio is not None and volratio>=1.5 and p>prev.Close,
          vol_high,"상승과 거래량 증가")
    award("거래량·OBV",volume_available and len(x)>=6 and ok("OBV20") and
          r.OBV>r.OBV20 and finite(x.OBV.iloc[-6]) and r.OBV>x.OBV.iloc[-6],
          obv_high,"OBV 상승")
    award("거래량·OBV",ok("RES") and p>r.RES,breakout,"20일 고점 돌파")

    flow_date=None
    flow_status="미적용(미국 시장)"
    if domestic:
        if isinstance(flow,dict) and all(k in flow for k in ("기관","외국인")):
            try:
                f=flow["외국인"]; i=flow["기관"]
                # Check valid numeric 5-/20-session sums, including legitimate zero.
                vals=[float(f["5일"]),float(f["20일"]),float(i["5일"]),float(i["20일"])]
                if not all(np.isfinite(v) for v in vals):raise ValueError("non-finite")
                flow_date=flow.get("date")
                if flow_date is None or not 0 <= (pd.Timestamp(x.index[-1]).normalize()-pd.Timestamp(flow_date).normalize()).days <= 7:
                    raise ValueError("stale flow")
                scores["외국인·기관"]=0
                award("외국인·기관",vals[0]>0,5,"외국인 5일 순매수")
                award("외국인·기관",vals[1]>0,5,"외국인 20일 누적 순매수")
                award("외국인·기관",vals[2]>0,5,"기관 5일 순매수")
                award("외국인·기관",vals[3]>0,5,"기관 20일 누적 순매수")
                flow_status="정상"
            except (TypeError,ValueError,KeyError):
                flow_status="자료 확인 불가"
        else:
            flow_status="자료 확인 불가"

    penalty=0
    if ok("RSI") and r.RSI>75:penalty+=10;details.append("RSI 과열")
    if rsi_delta is not None and rsi_delta<=-5:penalty+=5;details.append("RSI 둔화")
    if ok("BBUP") and p>r.BBUP:penalty+=5;details.append("볼린저 상단 추격 주의")
    if ok("ADX") and r.ADX>=25 and ok("PDI") and ok("MDI") and r.MDI>r.PDI:
        penalty+=15;details.append("ADX 강한 하락 추세")

    technical=sum(value for key,value in scores.items() if key!="외국인·기관" and value is not None)
    tech_max=80 if domestic else 100
    technical_net=max(0,technical-penalty)
    ready=len(x)>=200 and all(ok(k) for k in ("MA200","MA120","MACDH","ADX","OBV20")) and volume_available
    full_ready=ready and (not domestic or scores["외국인·기관"] is not None)
    technical_score=min(tech_max,technical_net) if ready else None
    # When investor flow is unavailable, show a clearly labelled 100-point
    # normalized TECHNICAL score, never silently count missing flows as zero.
    if full_ready:
        score=max(0,min(100,technical+int(scores.get("외국인·기관") or 0)-penalty))
        score_basis="종합 100점 (기술+수급)" if domestic else "미국 기술 100점"
    elif ready and domestic:
        score=round(100*technical_score/80)
        score_basis="기술환산 100점 (수급 제외)"
    else:
        score=None
        score_basis="시세/거래량 부족"
    if not ready:
        signal="데이터 부족"
    elif domestic and not full_ready:
        signal=("기술매수 후보·수급 미확인" if score>=60 else "관찰·수급 미확인")
    else:
        signal="강한 매수 검토" if score>=80 else "매수 검토" if score>=60 else "관찰"
    if ready and ok("RSI") and r.RSI>75:signal="과열·추격 주의"
    if ready and ok("MA20") and p<r.MA20 and ok("MACDH") and r.MACDH<0:signal="축소·매도 검토"
    if ready and ok("MA200") and p<r.MA200 and r.MA20<r.MA60:signal="매수 보류"
    if ready and ok("ADX") and r.ADX>=25 and ok("PDI") and ok("MDI") and r.MDI>r.PDI and p<r.MA20:
        signal="하락 추세·매수 보류"

    # Targets are distinct concepts:
    # resistance targets = previously observed levels ABOVE the latest close;
    # 2R = risk-management scenario, NOT predicted future share price.
    stop=target=None
    if ok("ATR") and r.ATR>0:
        stop=max(0.01,min(float(r.MA20),p-2*r.ATR) if ok("MA20") and p>r.MA20 else p-2*r.ATR)
        if stop<p:target=p+2*(p-stop)

    target_eval=integrated_targets(x,market,flow,stop)
    target1=target_eval["target1"]
    target2=target_eval["target2"]
    target1_basis=target_eval["target1_basis"]
    target2_basis=target_eval["target2_basis"]
    target1_r=target_eval["target1_r"]
    target2_r=target_eval["target2_r"]
    technical_signal_before_gate=signal
    signal,upside_pct,entry_rr,entry_reason=apply_target_risk_gate(
        signal,p,stop,target_eval["integrated_target"])
    return {"date":str(pd.Timestamp(x.index[-1]).date()),"price":p,
        "change":(p/float(prev.Close)-1)*100,"score":score,"technical_score":technical_score,
        "technical_max":tech_max,"score_basis":score_basis,"signal":signal,
        "technical_signal_before_gate":technical_signal_before_gate,
        "reason":" · ".join(details),"entry_reason":entry_reason,
        "upside_pct":upside_pct,"entry_rr":entry_rr,
        "scores":scores,"penalty":penalty,"flow_status":flow_status,"flow_date":flow_date,
        "rsi":num(r.RSI),"rsi_delta5":rsi_delta,
        "rsi_direction":("상승" if rsi_delta>=3 else "하락" if rsi_delta<=-3 else "횡보") if rsi_delta is not None else "데이터 부족",
        "volume_available":volume_available,"volume_now":int(r.Volume) if volume_available else None,
        "volume_avg20":num(r.VOL20) if volume_available else None,
        "adx":num(r.ADX),"pdi":num(r.PDI),"mdi":num(r.MDI),
        "obv_up":bool(volume_available and ok("OBV20") and r.OBV>r.OBV20),
        "bb_upper":num(r.BBUP),"bb_lower":num(r.BBLOW),
        "hi52":num(r.H52) if len(x)>=252 else None,
        "lo52":num(r.L52) if len(x)>=252 else None,"ma20":num(r.MA20),
        "ma60":num(r.MA60),"ma120":num(r.MA120),"ma200":num(r.MA200),
        "volratio":volratio,"support":num(r.SUP),"resistance":num(r.RES),
        "stop":stop,"target":target,"target1":target1,"target2":target2,
        "target1_basis":target1_basis,"target2_basis":target2_basis,
        "target1_r":target1_r,"target2_r":target2_r,
        "integrated_target":target_eval["integrated_target"],
        "target_grade":target_eval["target_grade"],
        "target_evidence":target_eval["target_evidence"],
        "target_r":target_eval["target_r"],
        "target_signal_points":target_eval["target_signal_points"],
        "resistance_breakout":target_eval["resistance_breakout"],
        "volume_profile":target_eval["volume_profile"],
        "target_inputs":target_eval["target_inputs"],
        "days":len(x)}
