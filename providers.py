import datetime as dt
import pandas as pd
from engine import normalize

def get_prices(code, market):
    import FinanceDataReader as fdr
    start=(dt.date.today()-dt.timedelta(days=850)).isoformat()
    failures=[]
    candidates=[f"NAVER:{code}",code] if market=="KR" else [f"STOOQ:{code}",code]
    for symbol in candidates:
        try:
            return normalize(fdr.DataReader(symbol,start)),f"FinanceDataReader {symbol}"
        except Exception as e:
            failures.append(f"{symbol}: {str(e)[:110]}")
    try:
        import yfinance as yf
        suffix=".KQ" if code in ("247540","086520","035900") else ".KS"
        ticker=code if market=="US" else code+suffix
        data=yf.download(ticker,period="2y",interval="1d",auto_adjust=True,progress=False,threads=False,timeout=12)
        return normalize(data),f"Yahoo {ticker}"
    except Exception as e:
        failures.append(f"Yahoo: {str(e)[:110]}")
    raise RuntimeError(" | ".join(failures))

def get_flows(code,end_date):
    try:
        from pykrx import stock
        end=pd.Timestamp(end_date).strftime("%Y%m%d")
        start=(pd.Timestamp(end_date)-pd.Timedelta(days=75)).strftime("%Y%m%d")
        data=stock.get_market_trading_volume_by_date(start,end,code,on="순매수")
        if data is None or data.empty:raise ValueError("KRX 데이터 없음")
        data=data.sort_index()
        ins=next((c for c in data.columns if "기관합계" in str(c)),None)
        foreign=next((c for c in data.columns if "외국인합계" in str(c) or str(c)=="외국인"),None)
        if not ins or not foreign:raise ValueError("기관 또는 외국인 수급 항목 없음")
        data=data[[ins,foreign]].apply(pd.to_numeric,errors="coerce").dropna()
        if data.empty:raise ValueError("순매수 데이터 없음")
        if len(data)<20:raise ValueError(f"20거래일 수급 자료 부족: {len(data)}일")
        def stat(s):
            streak=0
            for v in s.iloc[::-1]:
                if v>0:streak+=1
                else:break
            return {"1일":int(s.iloc[-1]),"5일":int(s.tail(5).sum()),"20일":int(s.tail(20).sum()),"연속 순매수일":streak}
        return {"date":str(pd.Timestamp(data.index[-1]).date()),"기관":stat(data[ins]),"외국인":stat(data[foreign])}
    except Exception as e:raise RuntimeError(f"KRX 순매수 조회 불가: {e}") from e
