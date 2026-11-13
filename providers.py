import datetime as dt
import pandas as pd
from engine import normalize

def get_prices(code, market):
    import FinanceDataReader as fdr
    start=(dt.date.today()-dt.timedelta(days=850)).isoformat()
    failures=[]
    candidates=[f"NAVER:{code}",code] if market in ("KR","KQ") else [f"STOOQ:{code}",code]
    for symbol in candidates:
        try:
            return normalize(fdr.DataReader(symbol,start)),f"FinanceDataReader {symbol}"
        except Exception as e:
            failures.append(f"{symbol}: {str(e)[:110]}")
    try:
        import yfinance as yf
        suffix=".KQ" if market=="KQ" else ".KS"
        ticker=code if market=="US" else code+suffix
        # Keep raw close series (not dividend-adjusted total-return proxy), consistent with
        # the displayed exchange price; critical for distribution-paying ETFs.
        data=yf.download(ticker,period="2y",interval="1d",auto_adjust=False,progress=False,threads=False,timeout=12)
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

def _parse_naver_investor_rows(html):
    """Parse the institution/foreign daily net-buy share volume table."""
    import re
    from bs4 import BeautifulSoup
    soup=BeautifulSoup(html,"html.parser")
    candidates=[]
    for table in soup.find_all("table"):
        headers=" ".join(th.get_text(" ",strip=True) for th in table.find_all("th"))
        if not all(v in headers for v in ("기관","외국인","날짜")):
            continue
        rows=[]
        for tr in table.find_all("tr"):
            cells=tr.find_all("td",recursive=False)
            if len(cells)<7:continue
            values=[cell.get_text(" ",strip=True).replace(",","").replace(" ","") for cell in cells]
            if not re.fullmatch(r"20\d{2}\.\d{2}\.\d{2}",values[0]):continue
            try:
                date=pd.to_datetime(values[0],format="%Y.%m.%d")
                # date/close/diff/change/volume/institution net/foreign net/foreign ownership
                rows.append((date,int(values[5]),int(values[6])))
            except (ValueError,TypeError,IndexError):continue
        if rows:candidates.append(rows)
    return max(candidates,key=len) if candidates else []

def _get_naver_flows(code):
    import requests
    session=requests.Session()
    session.headers.update({
        "User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0 Safari/537.36",
        "Referer":"https://finance.naver.com/",
        "Accept-Language":"ko-KR,ko;q=0.9,en-US;q=0.7"})
    rows=[]
    diagnostics=[]
    for page in range(1,5):
        try:
            response=session.get("https://finance.naver.com/item/frgn.naver",
                                 params={"code":code,"page":page},timeout=10)
            response.raise_for_status()
            response.encoding="euc-kr"
            parsed=_parse_naver_investor_rows(response.text)
            diagnostics.append(f"p{page}:{response.status_code}/{len(parsed)}일/{len(response.content)}B")
            rows.extend(parsed)
            if len({d for d,_,_ in rows})>=20:break
        except requests.RequestException as exc:
            diagnostics.append(f"p{page}:{type(exc).__name__} {str(exc)[:75]}")
            break
    if len({d for d,_,_ in rows})<20:
        raise ValueError("네이버 20거래일 조회 실패 ["+"; ".join(diagnostics)+"]")
    frame=pd.DataFrame(rows,columns=["date","기관","외국인"]).drop_duplicates("date").set_index("date")
    return _summarize_flow(frame,"기관","외국인","네이버 증권 일별 순매매 주수")

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

from functools import lru_cache
import time
from io import StringIO

@lru_cache(maxsize=3)
def _public_flow_snapshot(three_hour_slot):
    """CC BY 4.0 derived settled KRX/KIS share-flow aggregates; one fetch per 3 hours."""
    import requests
    url="https://raw.githubusercontent.com/james-brand/korea-market-data/main/data/foreign-flows-aggregates.csv"
    response=requests.get(url,timeout=12,headers={"User-Agent":"StockRadar200/1.0"})
    response.raise_for_status()
    if len(response.content)<10000:raise ValueError("수급 집계 파일 크기 이상")
    first=response.text.splitlines()[0]
    import re
    date_match=re.search(r"\bas of (\d{4}-\d{2}-\d{2})\b",first,flags=re.I)
    if not date_match:raise ValueError("수급 파일 기준일을 확인할 수 없음")
    as_of=date_match.group(1)
    age=(dt.date.today()-dt.date.fromisoformat(as_of)).days
    if age<0 or age>7:raise ValueError(f"수급 파일 기준일 {as_of} (현재 대비 {age}일) - 오래된 자료")
    df=pd.read_csv(StringIO(response.text),comment="#",dtype={"ticker":str})
    required={"ticker","f_1d_sh","f_5d_sh","f_20d_sh","i_1d_sh","i_5d_sh","i_20d_sh"}
    if not required.issubset(df.columns):raise ValueError(f"수급 컬럼 누락: {sorted(required-set(df.columns))}")
    if len(df)<1000:raise ValueError(f"수급 종목이 비정상적으로 적음: {len(df)}")
    df["ticker"]=df["ticker"].astype(str).str.strip().str.zfill(6)
    for col in required-{"ticker"}:
        df[col]=pd.to_numeric(df[col],errors="coerce")
    frame=df.set_index("ticker")
    return frame,as_of

def _get_public_aggregated_flows(code):
    frame,as_of=_public_flow_snapshot(int(time.time()//10800))
    code=str(code).zfill(6)
    if code not in frame.index:
        raise ValueError(f"종목 {code}은 제3자 수급집계 대상 밖(ETF 등)")
    row=frame.loc[code]
    mapping={"기관":("i_1d_sh","i_5d_sh","i_20d_sh"),
             "외국인":("f_1d_sh","f_5d_sh","f_20d_sh")}
    result={"date":as_of,
            "source":"K-Export Stars/KIS 기반 일별 집계 (CC BY 4.0)",
            "source_url":"https://github.com/james-brand/korea-market-data"}
    for label,cols in mapping.items():
        nums=[row[col] for col in cols]
        if not all(pd.notna(v) for v in nums):raise ValueError(f"{label}: 1·5·20일 수급 누락")
        result[label]={"1일":int(nums[0]),"5일":int(nums[1]),"20일":int(nums[2]),
                       "연속 순매수일":None}
    return result

def get_flows(code,end_date):
    """Use verified settled aggregate first, fall back to KRX when available."""
    if str(code) in ("472150","0262B0"):
        raise ValueError("커버드콜 ETF는 공개 투자자별 수급 집계 대상이 아닙니다. 기술점수만 제공합니다.")
    errors=[]
    # finance.naver.com/item/frgn.naver now redirects to stock.naver.com, so
    # the old HTML table is gone; do not burn ~40 seconds scraping empty pages.
    for provider in (_get_public_aggregated_flows,lambda symbol:_get_krx_flows(symbol,end_date)):
        try:
            result=provider(code)
            gap=(pd.Timestamp(end_date)-pd.Timestamp(result["date"])).days
            if gap<0 or gap>7:
                raise ValueError(f"가격일 {end_date} 대비 수급 기준일 {result['date']} ({gap}일 차이)")
            return result
        except Exception as exc:
            errors.append(f"{'공개 집계' if provider==_get_public_aggregated_flows else 'KRX'}: {str(exc)[:165]}")
    import logging
    logging.getLogger("stock_radar.flows").warning(
        "Investor flow failed %s %s: %s",code,end_date," | ".join(errors))
    raise RuntimeError(" | ".join(errors))
