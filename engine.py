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
    x=x[(x.Close>0)&(x.High>=x.Low)&(x.Volume>=0)]
    x=x.loc[~x.index.duplicated(keep="last")].sort_index()
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
    domestic=market=="KR"
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
                if flow_date is None or abs((pd.Timestamp(x.index[-1]).normalize()-pd.Timestamp(flow_date).normalize()).days)>7:
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
        signal="기술점수만·수급 미반영"
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

    observed=x.iloc[:-1].tail(252)
    levels=[]
    if len(observed)>=20:
        for period in (20,60,120,252):
            if len(observed)>=period:
                value=observed.High.iloc[-period:].max()
                if finite(value):
                    levels.append((float(value),f"직전 {period}일 고점"))
        # Local swing highs based only on historical candles: 3 trading sessions on
        # each side. They complement broad period highs where those overlap.
        highs=observed.High.astype(float)
        if len(highs)>=7:
            left=highs.shift(1).rolling(3,min_periods=3).max()
            right=highs.iloc[::-1].shift(1).rolling(3,min_periods=3).max().iloc[::-1]
            peaks=observed.loc[(highs>left)&(highs>=right)].tail(24)
            for _,bar in peaks.iterrows():
                value=float(bar.High)
                if finite(value):
                    levels.append((value,"과거 국소 고점"))
    # Reject levels at/below today's price. Nearby highs are one resistance zone;
    # don't show two virtually identical levels as 1st and 2nd targets.
    minimum_gap=max(p*0.005,float(r.ATR)*0.25 if ok("ATR") and r.ATR>0 else 0.0)
    usable=sorted((price,label) for price,label in levels if price>p+minimum_gap)
    distinct=[]
    for price,label in usable:
        if not distinct or price-distinct[-1][0]>=minimum_gap:
            distinct.append((price,label))
    target1=distinct[0][0] if distinct else None
    target2=distinct[1][0] if len(distinct)>1 else None
    target1_basis=distinct[0][1] if distinct else None
    target2_basis=distinct[1][1] if len(distinct)>1 else None
    target1_r=(target1-p)/(p-stop) if target1 is not None and stop is not None and p>stop else None
    target2_r=(target2-p)/(p-stop) if target2 is not None and stop is not None and p>stop else None
    return {"date":str(pd.Timestamp(x.index[-1]).date()),"price":p,
        "change":(p/float(prev.Close)-1)*100,"score":score,"technical_score":technical_score,
        "technical_max":tech_max,"score_basis":score_basis,"signal":signal,"reason":" · ".join(details),
        "scores":scores,"penalty":penalty,"flow_status":flow_status,"flow_date":flow_date,
        "rsi":num(r.RSI),"rsi_delta5":rsi_delta,
        "rsi_direction":("상승" if rsi_delta>=3 else "하락" if rsi_delta<=-3 else "횡보") if rsi_delta is not None else "데이터 부족",
        "volume_available":volume_available,"volume_now":int(r.Volume) if volume_available else None,
        "volume_avg20":num(r.VOL20) if volume_available else None,
        "adx":num(r.ADX),"pdi":num(r.PDI),"mdi":num(r.MDI),
        "obv_up":bool(volume_available and ok("OBV20") and r.OBV>r.OBV20),
        "bb_upper":num(r.BBUP),"bb_lower":num(r.BBLOW),
        "hi52":num(r.H52),"lo52":num(r.L52),"ma20":num(r.MA20),
        "ma60":num(r.MA60),"ma120":num(r.MA120),"ma200":num(r.MA200),
        "volratio":volratio,"support":num(r.SUP),"resistance":num(r.RES),
        "stop":stop,"target":target,"target1":target1,"target2":target2,
        "target1_basis":target1_basis,"target2_basis":target2_basis,
        "target1_r":target1_r,"target2_r":target2_r,"days":len(x)}
