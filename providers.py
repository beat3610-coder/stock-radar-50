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

def _summarize_flow(data, ins, foreign, source):
    data=data.sort_index()
    data=data[[ins,foreign]].apply(pd.to_numeric,errors="coerce").dropna()
    if len(data)<20:raise ValueError(f"{source}: 20거래일 중 {len(data)}일만 확인")
    def stat(s):
        streak=0
        for v in s.iloc[::-1]:
            if v>0:streak+=1
            else:break
        return {"1일":int(s.iloc[-1]),"5일":int(s.tail(5).sum()),
                "20일":int(s.tail(20).sum()),"연속 순매수일":streak}
    return {"date":str(pd.Timestamp(data.index[-1]).date()),
            "기관":stat(data[ins]),"외국인":stat(data[foreign]),"source":source}

def _get_naver_flows(code):
    """Public Naver Finance investor table, verified against visible 005930 layout.
    Two pages (normally 20 entries/page), require 20 distinct valid trading dates.
    """
    import requests
    from bs4 import BeautifulSoup
    rows=[]
    session=requests.Session()
    session.headers.update({"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124.0 Safari/537.36",
                            "Referer":"https://finance.naver.com/"})
    for page in (1,2,3):
        url=f"https://finance.naver.com/item/frgn.naver?code={code}&page={page}"
        response=session.get(url,timeout=7)
        response.raise_for_status()
        response.encoding=response.apparent_encoding
        soup=BeautifulSoup(response.text,"html.parser")
        found=0
        for tr in soup.select("table tr"):
            cells=tr.find_all("td",recursive=False)
            if len(cells)<8:continue
            raw=[cell.get_text(" ",strip=True).replace(",","").replace(" ","") for cell in cells]
            try:
                when=pd.to_datetime(raw[0],format="%Y.%m.%d",errors="raise")
                ins=int(raw[5]);foreign=int(raw[6])
                rows.append((when,ins,foreign));found+=1
            except (ValueError,IndexError,TypeError):continue
        if len({row[0] for row in rows})>=20:break
        if found==0:raise ValueError(f"네이버 투자자 수급 표 파싱 실패(page {page}; HTML 구조 확인 필요)")
    if not rows:raise ValueError("네이버 수급 데이터 없음")
    frame=pd.DataFrame(rows,columns=["date","기관","외국인"]).drop_duplicates("date").set_index("date")
    return _summarize_flow(frame,"기관","외국인","네이버 증권 투자자별 일별 순매매")

def _get_krx_flows(code,end_date):
    from pykrx import stock
    end=pd.Timestamp(end_date).strftime("%Y%m%d")
    start=(pd.Timestamp(end_date)-pd.Timedelta(days=95)).strftime("%Y%m%d")
    data=stock.get_market_trading_volume_by_date(start,end,code,on="순매수")
    if data is None or data.empty:raise ValueError("KRX 수급 데이터 없음")
    ins=next((col for col in data.columns if "기관합계" in str(col)),None)
    foreign=next((col for col in data.columns if "외국인합계" in str(col) or str(col)=="외국인"),None)
    if not ins or not foreign:raise ValueError(f"KRX 투자자별 컬럼 누락: {list(data.columns)}")
    return _summarize_flow(data,ins,foreign,"KRX pykrx 투자자별 순매매")

def get_flows(code,end_date):
    """Try visible retail market data first, then KRX; require aligned latest date."""
    errors=[]
    for provider in (_get_naver_flows,lambda symbol:_get_krx_flows(symbol,end_date)):
        try:
            result=provider(code)
            gap=abs((pd.Timestamp(end_date)-pd.Timestamp(result["date"])).days)
            if gap>7:
                raise ValueError(f"가격일 {end_date} 대비 수급 기준일 {result['date']} ({gap}일 차이)")
            return result
        except Exception as exc:
            errors.append(f"{'네이버' if provider==_get_naver_flows else 'KRX'}: {str(exc)[:165]}")
    raise RuntimeError(" | ".join(errors))
