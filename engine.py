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
    if len(x)<25: raise ValueError(f"자료 부족: {len(x)}거래일")
    return x

def compute(data):
    x=normalize(data)
    c=x.Close
    for n in (5,20,60,200): x[f"MA{n}"]=c.rolling(n).mean()
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
    return x

def finite(v): return pd.notna(v) and bool(np.isfinite(v))

def evaluate(x):
    r=x.iloc[-1]; prev=x.iloc[-2]; p=float(r.Close)
    score=0; details=[]
    def add(cond,points,label):
        nonlocal score
        if cond: score+=points; details.append(label)
    add(finite(r.MA20) and p>r.MA20,10,"20일선 위")
    add(finite(r.MA60) and r.MA20>r.MA60,15,"20일선 > 60일선")
    add(finite(r.MA200) and r.MA60>r.MA200,15,"60일선 > 200일선")
    add(finite(r.MA200) and p>r.MA200,10,"200일선 위")
    add(finite(r.MACDH) and r.MACDH>0,10,"MACD 상승")
    add(finite(r.RSI) and 45<=r.RSI<=68,10,"RSI 상승 구간")
    add(finite(r.VOL20) and r.VOL20>0 and r.Volume>1.5*r.VOL20 and p>prev.Close,10,"거래량 동반 상승")
    add(finite(r.RES) and p>r.RES,10,"20일 고점 돌파")
    add(len(x)>25 and finite(x.MA20.iloc[-6]) and r.MA20>x.MA20.iloc[-6],10,"20일선 상승")
    if finite(r.RSI) and r.RSI>75: score-=15; details.append("RSI 과열")
    if finite(r.MA200) and p<r.MA200:score-=10;details.append("200일선 하회")
    ready=len(x)>=200 and finite(r.MA200) and finite(r.MACDH)
    signal="데이터 부족" if not ready else ("강한 매수 검토" if score>=80 else "매수 검토" if score>=60 else "관찰")
    if ready and r.RSI>75: signal="과열·추격 주의"
    if ready and p<r.MA20 and r.MACDH<0: signal="축소·매도 검토"
    if ready and p<r.MA200 and r.MA20<r.MA60:signal="매수 보류"
    stop=target=None
    if finite(r.ATR) and r.ATR>0:
        stop=max(0.01,min(float(r.MA20),p-2*r.ATR) if finite(r.MA20) and p>r.MA20 else p-2*r.ATR)
        if stop<p:target=p+2*(p-stop)
    def num(v):return float(v) if finite(v) else None
    return {"date":str(pd.Timestamp(x.index[-1]).date()),"price":p,"change":(p/float(prev.Close)-1)*100,
       "score":max(0,min(100,score)) if ready else None,"signal":signal,"reason":" · ".join(details),
       "rsi":num(r.RSI),"hi52":num(r.H52),"lo52":num(r.L52),"ma20":num(r.MA20),
       "ma60":num(r.MA60),"ma200":num(r.MA200),"volratio":num(r.Volume/r.VOL20) if finite(r.VOL20) and r.VOL20>0 else None,
       "support":num(r.SUP),"resistance":num(r.RES),"stop":stop,"target":target,"days":len(x)}
