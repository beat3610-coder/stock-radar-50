import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import time
import pandas as pd
from engine import normalize

PRICE_CACHE = Path(os.environ.get("STOCK_RADAR_PRICE_CACHE", ".cache/stock-radar/prices"))
RETRY_DELAYS = (0.5, 1.0)

def _cache_path(code, market, start):
    key=hashlib.sha256(f"{market}:{code}:{start}".encode()).hexdigest()[:24]
    return PRICE_CACHE/f"{key}.csv"

def _read_cached(path, min_bars, max_age_days):
    if not path.exists():return None
    try:
        frame=normalize(pd.read_csv(path,index_col=0,parse_dates=True))
        last=pd.Timestamp(frame.index[-1]).date()
        age=(dt.date.today()-last).days
        if len(frame)<min_bars or age<0 or age>max_age_days:return None
        return frame
    except Exception:
        return None

def _write_cached(path, frame):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(".tmp")
    frame.to_csv(tmp,index_label="Date")
    tmp.replace(path)

def get_prices(code, market, *, start_date=None, min_bars=None):
    """Live default stays compact; research may ask for longer raw OHLCV history.

    The provider may return less history than requested; caller must inspect
    the actual first/last dates and observations rather than assume coverage.
    """
    start=(dt.date.today()-dt.timedelta(days=850)).isoformat() if start_date is None else str(pd.Timestamp(start_date).date())
    required_bars=(250 if start_date is not None else 50) if min_bars is None else int(min_bars)
    cache_path=_cache_path(code,market,start)
    cached=_read_cached(cache_path,required_bars,14)
    if cached is not None:
        return cached,f"validated local OHLCV cache ({len(cached)} bars; through {cached.index[-1].date()})"
    # A validated cache fetched from an earlier start date contains every bar
    # needed by a later audit window. Reuse it instead of forcing a duplicate
    # provider download under a different cache key.
    if start_date is not None:
        for prior_start in ("2006-01-01","2010-01-01","2016-01-01"):
            if prior_start>=start:continue
            prior_path=_cache_path(code,market,prior_start)
            prior=_read_cached(prior_path,required_bars,14)
            if prior is not None:
                prior=prior.loc[pd.Timestamp(start):]
                if len(prior)>=required_bars:
                    _write_cached(cache_path,prior)
                    return prior,f"validated reused OHLCV cache from {prior_start} ({len(prior)} bars; through {prior.index[-1].date()})"
    failures=[]
    try:
        import FinanceDataReader as fdr
        candidates=[f"NAVER:{code}",code] if market in ("KR","KQ") else [f"STOOQ:{code}",code]
        for symbol in candidates:
            for attempt in range(1,len(RETRY_DELAYS)+2):
                try:
                    frame=normalize(fdr.DataReader(symbol,start))
                    if len(frame)<required_bars:raise ValueError(f"history too short: {len(frame)}<{required_bars}")
                    _write_cached(cache_path,frame)
                    return frame,f"FinanceDataReader {symbol} ({len(frame)} bars)"
                except Exception as e:
                    failures.append(f"{symbol} try {attempt}: {str(e)[:95]}")
                    if attempt<=len(RETRY_DELAYS):time.sleep(RETRY_DELAYS[attempt-1])
    except Exception as e:
        failures.append(f"FinanceDataReader unavailable: {str(e)[:95]}")
    try:
        import yfinance as yf
        suffix=".KQ" if market=="KQ" else ".KS"
        ticker=code if market=="US" else code+suffix
        for attempt in range(1,len(RETRY_DELAYS)+2):
            try:
                data=yf.download(ticker,period="2y" if start_date is None else None,
                                 start=start if start_date is not None else None,
                                 interval="1d",auto_adjust=False,progress=False,
                                 threads=False,timeout=20)
                frame=normalize(data)
                if len(frame)<required_bars:raise ValueError(f"history too short: {len(frame)}<{required_bars}")
                _write_cached(cache_path,frame)
                return frame,f"Yahoo {ticker} ({len(frame)} bars)"
            except Exception as e:
                failures.append(f"Yahoo {ticker} try {attempt}: {str(e)[:95]}")
                if attempt<=len(RETRY_DELAYS):time.sleep(RETRY_DELAYS[attempt-1])
    except Exception as e:
        failures.append(f"Yahoo unavailable: {str(e)[:95]}")
    # A stale complete cache can still support fixed historical windows, but its
    # as-of date is returned explicitly so callers can reject incomplete windows.
    try:
        frame=normalize(pd.read_csv(cache_path,index_col=0,parse_dates=True))
        if len(frame)>=required_bars:
            return frame,f"stale historical cache ({len(frame)} bars; through {frame.index[-1].date()})"
    except Exception:pass
    raise RuntimeError(" | ".join(failures))

def get_historical_prices(code,market,*,start_date="2006-01-01",min_bars=250):
    """Explicit research-only historical fetch, never a silent 850-day cap."""
    return get_prices(code,market,start_date=start_date,min_bars=min_bars)


MARKET_SERIES={
    # FinanceDataReader symbols are preferred; Yahoo symbols are explicit fallback.
    "KOSPI":{"fdr":"KS11","yahoo":"^KS11"},
    "KOSDAQ":{"fdr":"KQ11","yahoo":"^KQ11"},
    "SP500":{"fdr":"US500","yahoo":"^GSPC"},
    "SOX":{"fdr":"^SOX","yahoo":"^SOX"},
    "VIX":{"fdr":"^VIX","yahoo":"^VIX"},
    "USDKRW":{"fdr":"USD/KRW","yahoo":"KRW=X"},
}

def _normalize_market_close(data):
    """Normalize index/FX series without applying stock-candle OHLC checks."""
    if data is None or data.empty:raise ValueError("market-series response is empty")
    frame=data.copy()
    if isinstance(frame.columns,pd.MultiIndex):frame.columns=frame.columns.get_level_values(0)
    frame.columns=[str(c).strip().title() for c in frame.columns]
    close_key=next((k for k in ("Close","Adj Close","Value") if k in frame.columns),None)
    if close_key is None:
        if frame.shape[1]==1:close_key=frame.columns[0]
        else:raise ValueError(f"market-series Close column missing: {list(frame.columns)}")
    close=pd.to_numeric(frame[close_key],errors="coerce").dropna()
    close=close[close>0]
    close=close.loc[~close.index.duplicated(keep="last")].sort_index()
    if len(close)<2:raise ValueError(f"market-series close history too short: {len(close)}")
    return pd.DataFrame({"Close":close})

def get_market_series_bundle(*,start_date="2016-01-01",end_date=None):
    """Fetch/cache daily market series; report every source failure explicitly."""
    end=(dt.date.today()+dt.timedelta(days=1)).isoformat() if end_date is None else str(pd.Timestamp(end_date).date())
    root=PRICE_CACHE.parent/"indices"
    root.mkdir(parents=True,exist_ok=True)
    frames={};diagnostics={}
    import logging
    logger=logging.getLogger("stock_radar.market_series")
    for key,spec in MARKET_SERIES.items():
        path=root/f"{key.lower()}_{start_date}_{end}.csv"
        attempts=[]
        frame=None;source=None;loaded_cache=False
        if path.exists():
            try:
                cached=_normalize_market_close(pd.read_csv(path,index_col=0,parse_dates=True))
                if not cached.empty and cached.index.min()<=pd.Timestamp(start_date) and cached.index.max()>=pd.Timestamp(end)-pd.Timedelta(days=12):
                    metadata_path=path.with_suffix(".source.json")
                    metadata=json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
                    cached_source=metadata.get("source")
                    if cached_source:
                        frame=cached;source=cached_source;loaded_cache=True
                    else:
                        # Earlier caches did not persist provider provenance;
                        # refresh once so the audit never labels a cached value
                        # with a stale source from the prior loop iteration.
                        attempts.append("cache source metadata missing; refreshing from declared providers")
            except Exception as exc:
                attempts.append(f"cache: {type(exc).__name__}: {str(exc)[:120]}")
        if frame is None:
            try:
                import FinanceDataReader as fdr
                for attempt in range(1,4):
                    try:
                        candidate=_normalize_market_close(fdr.DataReader(spec["fdr"],start_date,end))
                        frame=candidate
                        source=f"FinanceDataReader {spec['fdr']}"
                        break
                    except Exception as exc:
                        attempts.append(f"FDR {spec['fdr']} try {attempt}: {type(exc).__name__}: {str(exc)[:120]}")
                        if attempt<3:time.sleep(RETRY_DELAYS[min(attempt-1,len(RETRY_DELAYS)-1)])
            except Exception as exc:
                attempts.append(f"FinanceDataReader unavailable: {type(exc).__name__}: {str(exc)[:120]}")
        if frame is None:
            try:
                import yfinance as yf
                for attempt in range(1,4):
                    try:
                        candidate=yf.download(spec["yahoo"],start=start_date,end=end,interval="1d",
                            auto_adjust=False,progress=False,threads=False,timeout=20)
                        if isinstance(candidate.columns,pd.MultiIndex):
                            candidate.columns=candidate.columns.get_level_values(0)
                        candidate=_normalize_market_close(candidate)
                        frame=candidate
                        source=f"Yahoo {spec['yahoo']}"
                        break
                    except Exception as exc:
                        attempts.append(f"Yahoo {spec['yahoo']} try {attempt}: {type(exc).__name__}: {str(exc)[:120]}")
                        if attempt<3:time.sleep(RETRY_DELAYS[min(attempt-1,len(RETRY_DELAYS)-1)])
            except Exception as exc:
                attempts.append(f"yfinance unavailable: {type(exc).__name__}: {str(exc)[:120]}")
        if frame is not None:
            frame=frame.sort_index()
            if not loaded_cache:
                frame.to_csv(path,index_label="Date")
                path.with_suffix(".source.json").write_text(json.dumps({"source":source},ensure_ascii=False),encoding="utf-8")
            frames[key]=frame
            diagnostics[key]={"status":"cache" if loaded_cache else "ok","source":source,"requested_symbol":spec,
                "first_date":str(frame.index.min().date()),"last_date":str(frame.index.max().date()),
                "observations":len(frame),"failures":attempts}
        else:
            diagnostics[key]={"status":"unavailable","source":None,"requested_symbol":spec,
                "first_date":None,"last_date":None,"observations":0,"failures":attempts}
            logger.error("Market series unavailable %s: %s",key," | ".join(attempts))
    return frames,diagnostics


def get_historical_investor_flows(code,*,start_date="2016-01-01",end_date=None):
    """Cached daily KRX investor net-buy share history; raises with source detail."""
    end=(dt.date.today()+dt.timedelta(days=1)).isoformat() if end_date is None else str(pd.Timestamp(end_date).date())
    cache_root=PRICE_CACHE.parent/"flows";cache_root.mkdir(parents=True,exist_ok=True)
    key=hashlib.sha256(f"{code}:{start_date}:{end}".encode()).hexdigest()[:24]
    path=cache_root/f"{key}.csv"
    if path.exists():
        try:
            cached=pd.read_csv(path,index_col=0,parse_dates=True)
            cached.index=pd.to_datetime(cached.index)
            cached=cached.apply(pd.to_numeric,errors="coerce").dropna(how="all")
            if not cached.empty:return cached,"validated local KRX investor-flow cache"
        except Exception:pass
    try:
        from pykrx import stock
        start=pd.Timestamp(start_date).strftime("%Y%m%d");finish=pd.Timestamp(end).strftime("%Y%m%d")
        data=stock.get_market_trading_volume_by_date(start,finish,str(code).zfill(6),on="순매수")
        if data is None or data.empty:raise ValueError("KRX returned an empty investor-flow frame")
        ins=next((col for col in data.columns if "기관합계" in str(col)),None)
        foreign=next((col for col in data.columns if "외국인합계" in str(col) or str(col)=="외국인"),None)
        if ins is None or foreign is None:
            raise ValueError(f"KRX investor-flow columns missing: {list(data.columns)}")
        frame=data[[ins,foreign]].copy();frame.columns=["institution_net_shares","foreign_net_shares"]
        frame.index=pd.to_datetime(frame.index);frame=frame.apply(pd.to_numeric,errors="coerce").dropna(how="all").sort_index()
        if len(frame)<250:raise ValueError(f"KRX daily-flow history too short: {len(frame)} observations")
        frame.to_csv(path,index_label="Date")
        return frame,"pykrx KRX daily investor net-buy shares"
    except Exception as exc:
        raise RuntimeError(f"historical KRX investor flow unavailable for {code}: {type(exc).__name__}: {str(exc)[:180]}") from exc


def verify_recent_close(code, market, primary_date, primary_price, primary_source,
                        tolerance_pct=0.5):
    """Compare latest closing *date and price* against Yahoo as a secondary feed.

    Agreement is never official KRX verification. Same-source comparisons are
    explicitly classified as unverified rather than artificially corroborated.
    """
    note={"status":"unverified","secondary_source":"Yahoo Finance",
          "reason":"다른 공급처 확정 종가와 아직 대조되지 않음",
          "tolerance_pct":tolerance_pct,"secondary_close":None,"difference_pct":None}
    if primary_source.startswith("Yahoo"):
        note["reason"]="주 시세와 보조 시세가 같은 Yahoo 원천"
        return note
    if market=="US" and "STOOQ:" not in primary_source.upper():
        note["reason"]="미국 주 시세와 Yahoo의 데이터 원천 독립성이 확인되지 않음"
        return note
    try:
        import yfinance as yf
        ticker=code if market=="US" else code+(".KQ" if market=="KQ" else ".KS")
        result=yf.download(ticker,period="10d",interval="1d",
                           auto_adjust=False,progress=False,
                           threads=False,timeout=12)
        candles=normalize(result)
        date=pd.Timestamp(candles.index[-1]).date()
        expected=pd.Timestamp(primary_date).date()
        if date!=expected:
            note["reason"]=f"기준 거래일 불일치: 주 {expected}, 보조 {date}"
            return note
        other=float(candles.Close.iloc[-1])
        expected_price=float(primary_price)
        if not 0<other or not 0<expected_price:
            note["reason"]="비정상적 보조 종가"
            return note
        diff=100*abs(expected_price/other-1)
        note.update(status=("agree" if diff<=tolerance_pct else "mismatch"),
                    reason=("동일 거래일 보조 종가 일치(거래소 공식 검증 아님)" if diff<=tolerance_pct
                            else "동일 거래일 종가 불일치 — 매매 판단 이용 금지"),
                    secondary_close=other,difference_pct=round(diff,4))
    except Exception as exc:
        note["reason"]=f"보조 시세 조회 불가: {str(exc)[:90]}"
    return note


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
