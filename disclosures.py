"""Free, unauthenticated DART disclosure search with resumable local cache."""
from __future__ import annotations

from datetime import date
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import re
import threading
import time
from urllib.parse import parse_qs, urlparse

import pandas as pd

DISCLOSURE_URL="https://dart.fss.or.kr/dsab007/detailSearch.ax"
DEFAULT_CACHE=Path(".cache/stock-radar/disclosures")
MAX_PAGES_PER_COMPANY_YEAR=30
MAX_RESULTS=100
REQUEST_INTERVAL_SECONDS=0.3
_REQUEST_RATE_LOCK=threading.Lock()
_NEXT_REQUEST_AT=0.0
BAD_KEYWORDS=("유상증자","전환사채권발행결정","신주인수권부사채","횡령","배임",
 "감사의견 거절","의견거절","관리종목 지정","상장폐지사유 발생",
 "불성실공시법인 지정","영업정지","채무불이행","최대주주 변경","최대주주변경",
 "경영권 분쟁","소송등의 제기","공급계약 해지")
BAD_TITLE_EXCLUSIONS=("소유주식변동신고서","임원ㆍ주요주주","주요주주","해외증권시장")
POSITIVE_KEYWORDS=("자기주식취득","자사주취득","자기주식 취득","단일판매·공급계약체결",
 "단일판매공급계약체결","무상증자")
PERIODIC_KEYWORDS=("사업보고서","반기보고서","분기보고서")


def bad_disclosure_keyword_matches(title):
    """Return predeclared bad-title matches, honoring explicit exclusions."""
    title=str(title or "")
    if any(term in title for term in BAD_TITLE_EXCLUSIONS):return ()
    return tuple(term for term in BAD_KEYWORDS if term in title)


def request_interval_seconds():
    try:value=float(os.environ.get("DISCLOSURE_REQUEST_INTERVAL_SECONDS",REQUEST_INTERVAL_SECONDS))
    except (TypeError,ValueError):value=REQUEST_INTERVAL_SECONDS
    return max(0.0,value)


class _DisclosureTableParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.rows=[];self.row=None;self.cell=None
    def handle_starttag(self,tag,attrs):
        attrs=dict(attrs)
        if tag=="tr":self.row=[]
        elif tag in ("td","th") and self.row is not None:self.cell={"text":[],"hrefs":[]}
        elif tag=="a" and self.cell is not None and attrs.get("href"):
            self.cell["hrefs"].append(attrs["href"])
    def handle_data(self,data):
        if self.cell is not None:self.cell["text"].append(data)
    def handle_endtag(self,tag):
        if tag in ("td","th") and self.row is not None and self.cell is not None:
            self.row.append({"text":" ".join(" ".join(self.cell["text"]).split()),"hrefs":self.cell["hrefs"]})
            self.cell=None
        elif tag=="tr" and self.row is not None:
            if self.row:self.rows.append(self.row)
            self.row=None;self.cell=None


def parse_disclosure_html(html):
    parser=_DisclosureTableParser();parser.feed(html or "")
    result=[]
    for cells in parser.rows:
        texts=[c["text"] for c in cells]
        joined=" ".join(texts)
        date_match=re.search(r"(20\d{2})[.\-/](\d{1,2})[.\-/](\d{1,2})",joined)
        if not date_match:continue
        receipt=f"{int(date_match.group(1)):04d}-{int(date_match.group(2)):02d}-{int(date_match.group(3)):02d}"
        rcp=None
        for cell in cells:
            for href in cell["hrefs"]:
                query=parse_qs(urlparse(href).query)
                vals=query.get("rcpNo",[])
                if vals:rcp=vals[0];break
                match=re.search(r"(?:rcpNo\s*[=:]\s*['\"]?|report\s*\()([0-9]{10,20})",href,re.I)
                if match:rcp=match.group(1);break
                match=re.search(r"([0-9]{14})",href)
                if match:rcp=match.group(1);break
            if rcp:break
        if rcp is None:
            match=re.search(r"\b([0-9]{14})\b",joined)
            if match:rcp=match.group(1)
        title=next((texts[i] for i in (2,1,0) if i<len(texts) and texts[i]),"")
        if not title:continue
        result.append({"receipt_date":receipt,"company":texts[1] if len(texts)>1 else "",
            "title":title,"submitter":texts[3] if len(texts)>3 else "","rcpNo":rcp})
    return result


def _read_cache(path):
    try:
        data=json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data,dict) and isinstance(data.get("records"),list):return data
    except (OSError,ValueError,TypeError):pass
    return None


def get_cached_company_disclosures(ticker,name,start_date="2015-12-01",end_date="2026-11-13",*,cache_dir=None):
    """Read the resumable cache without making network requests during an audit."""
    ticker=str(ticker).strip();name=str(name).strip()
    start=pd.Timestamp(start_date).normalize();end=pd.Timestamp(end_date).normalize()
    root=Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE
    cached=_read_cache(root/f"{ticker}.json")
    years=list(range(start.year,end.year+1))
    if not cached:
        reason="No warmup cache; use the manual disclosure-warmup workflow."
        return {"records":[],"diagnostics":{"ticker":ticker,"company_name":name,
            "source":"DART public HTML detailSearch.ax cache-only","requested_start":str(start.date()),
            "requested_end":str(end.date()),"first_date":None,"last_date":None,"rows_n":0,
            "requests_n":0,"pages_n":0,"attempted":False,"available":False,"partial":False,
            "year_coverage_pct":0.0,"available_years_n":0,"requested_years_n":len(years),
            "years_with_records":[],"complete":False,"cache_hit":False,
            "request_interval_seconds":request_interval_seconds(),"failures":[reason],"reason":reason}}
    cached_diag=cached.get("diagnostics",{})
    rows=[row for row in cached.get("records",[]) if row.get("receipt_date") and
        str(start.date())<=str(row["receipt_date"])<=str(end.date())]
    record_years=sorted({int(str(row["receipt_date"])[:4]) for row in rows
                         if str(row.get("receipt_date",""))[:4].isdigit()})
    available_years=sum(year in record_years for year in years)
    coverage=100.0*available_years/len(years) if years else 0.0
    year_status=cached_diag.get("year_status",{})
    if year_status:
        complete=all(bool(year_status.get(str(year),{}).get("complete")) for year in years)
    else:
        complete=bool(cached_diag.get("complete") and
            cached_diag.get("requested_start","9999-12-31")<=str(start.date()) and
            cached_diag.get("requested_end","0000-01-01")>=str(end.date()))
    available=bool(rows) and coverage>=80.0
    partial=bool(rows) and (not complete or available_years<len(years))
    failures=list(cached_diag.get("failures",[]))
    if not complete and not failures:failures=["Incomplete cached history; warmup will resume missing years/pages."]
    diag={**cached_diag,"ticker":ticker,"company_name":name,"source":"DART public HTML detailSearch.ax cache-only",
        "requested_start":str(start.date()),"requested_end":str(end.date()),
        "first_date":rows[0]["receipt_date"] if rows else None,
        "last_date":rows[-1]["receipt_date"] if rows else None,"rows_n":len(rows),
        "available":available,"partial":partial,"year_coverage_pct":round(coverage,2),
        "available_years_n":available_years,"requested_years_n":len(years),
        "years_with_records":record_years,"complete":complete,"cache_hit":True,
        "attempted":False,"request_interval_seconds":request_interval_seconds(),"failures":failures}
    return {"records":rows,"diagnostics":diag}


def _paced_get(session,params):
    """Start public DART requests at least 300 ms apart across all workers."""
    global _NEXT_REQUEST_AT
    with _REQUEST_RATE_LOCK:
        now=time.monotonic()
        wait=max(0.0,_NEXT_REQUEST_AT-now)
        if wait:time.sleep(wait)
        interval=request_interval_seconds()
        _NEXT_REQUEST_AT=time.monotonic()+interval
    return session.get(DISCLOSURE_URL,params=params,timeout=20)


def get_company_disclosures(ticker,name,start_date="2015-12-01",end_date=None,*,session=None,
                            cache_dir=None,force_refresh=False):
    """Return ``{"records": [...], "diagnostics": {...}}`` for one company.

    DART search is split into calendar years because its ten-year range query
    returns an empty result. Only the observed free HTML endpoint is used; no
    OpenDART key, KRX account, or paid service is required.
    """
    import requests
    ticker=str(ticker).strip();name=str(name).strip()
    start=pd.Timestamp(start_date).normalize();end=(pd.Timestamp(date.today()) if end_date is None else pd.Timestamp(end_date)).normalize()
    root=Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE
    root.mkdir(parents=True,exist_ok=True);path=root/f"{ticker}.json"
    cached=None if force_refresh else _read_cache(path)
    cached_diag=(cached or {}).get("diagnostics",{})
    if cached:
        if (cached_diag.get("complete") and cached_diag.get("requested_start","9999-12-31")<=str(start.date())
                and cached_diag.get("requested_end","0000-01-01")>=str(end.date())):
            records=[r for r in cached["records"] if str(start.date())<=r["receipt_date"]<=str(end.date())]
            years=list(range(start.year,end.year+1))
            record_years=sorted({int(str(row["receipt_date"])[:4]) for row in records
                                 if str(row.get("receipt_date",""))[:4].isdigit()})
            available_years=sum(year in record_years for year in years)
            coverage=100.0*available_years/len(years) if years else 0.0
            available=bool(records) and coverage>=80.0
            return {"records":records,"diagnostics":{**cached_diag,"cache_hit":True,"ticker":ticker,
                "available":available,"partial":bool(records) and available_years<len(years),
                "year_coverage_pct":round(coverage,2),"available_years_n":available_years,
                "requested_years_n":len(years),"years_with_records":record_years,
                "request_interval_seconds":request_interval_seconds()}}

    if session is None:session=requests.Session()
    try:session.headers.update({"User-Agent":"Mozilla/5.0 (compatible; stock-radar-research/1.0)","Accept":"text/html,application/xhtml+xml"})
    except Exception:pass
    records={r.get("rcpNo") or (r.get("receipt_date"),r.get("title")):r for r in (cached or {}).get("records",[])
             if r.get("receipt_date") and str(start.date())<=r["receipt_date"]<=str(end.date())}
    requested_years=list(range(start.year,end.year+1))
    year_status={str(year):dict(status) for year,status in cached_diag.get("year_status",{}).items()
                 if isinstance(status,dict)}
    prior_failures=list(cached_diag.get("failures",[]))
    legacy_failed_pages={};legacy_failed_years=set()
    if not year_status:
        for failure in prior_failures:
            years_found=re.findall(r"20\d{2}",str(failure))
            legacy_failed_years.update(int(value) for value in years_found)
            page_match=re.search(r"year=(20\d{2})\s+page=(\d+)",str(failure))
            if page_match:
                legacy_failed_pages.setdefault(int(page_match.group(1)),set()).add(int(page_match.group(2)))
    requests_n=0;failures=[];pages_n=0;attempt_log=[]
    cache_range_covers=(cached_diag.get("requested_start","9999-12-31")<=str(start.date())
                        and cached_diag.get("requested_end","0000-01-01")>=str(end.date()))
    for year in requested_years:
        lo=max(start,pd.Timestamp(year=year,month=1,day=1));hi=min(end,pd.Timestamp(year=year,month=12,day=31))
        key=str(year);state=year_status.get(key,{})
        state_covers=(not state.get("window_start") or
            (state.get("window_start")<=str(lo.date()) and state.get("window_end","0000-01-01")>=str(hi.date())))
        if state.get("complete") and state_covers:
            continue
        year_records=[r for r in records.values() if str(r.get("receipt_date",""))[:4]==key]
        if not state and year_records and cache_range_covers and year not in legacy_failed_years:
            # Older caches had no per-year cursor. Treat a populated, non-failed
            # year as complete and persist that fact so later runs can skip it.
            year_status[key]={"complete":True,"completed_pages":[],"failed_pages":[],
                "next_page":None,"window_start":str(lo.date()),"window_end":str(hi.date()),
                "inferred_from_cached_records":True,"rows_n":len(year_records)}
            continue
        completed_pages=set(int(x) for x in state.get("completed_pages",[]) if str(x).isdigit())
        failed_pages=set(int(x) for x in state.get("failed_pages",[]) if str(x).isdigit())
        failed_pages.update(legacy_failed_pages.get(year,set()))
        start_page=min(failed_pages) if failed_pages else int(state.get("next_page") or 1)
        if not state and year in legacy_failed_years and not failed_pages:
            start_page=1
        year_failure_pages=[];year_complete=False;next_page=start_page;year_requests_n=0
        for page in range(start_page,MAX_PAGES_PER_COMPANY_YEAR+1):
            params={"currentPage":page,"maxResults":MAX_RESULTS,"sort":"date","series":0,
                "textCrpNm":name,"startDate":lo.strftime("%Y%m%d"),"endDate":hi.strftime("%Y%m%d")}
            html=None
            for attempt in range(1,5):
                requests_n+=1;year_requests_n+=1
                try:
                    response=_paced_get(session,params)
                    if response.status_code in (429,500,502,503,504):
                        raise requests.HTTPError(f"HTTP {response.status_code}",response=response)
                    response.raise_for_status();html=response.text
                except Exception as exc:
                    entry=f"year={year} page={page} try={attempt}: {type(exc).__name__}: {str(exc)[:180]}"
                    attempt_log.append(entry)
                    if attempt==4:failures.append(entry)
                    else:time.sleep(float(2**(attempt-1)))
                if html is not None:break
            if html is None:
                year_failure_pages.append(page);next_page=page;break
            page_rows=parse_disclosure_html(html);pages_n+=1
            if not page_rows:
                completed_pages.add(page);year_complete=True;next_page=None;break
            before=len(records)
            for row in page_rows:
                if row["rcpNo"]:record_key=row["rcpNo"]
                else:record_key=(row["receipt_date"],row["title"],row.get("submitter"))
                records[record_key]=row
            completed_pages.add(page)
            if len(page_rows)<MAX_RESULTS:
                year_complete=True;next_page=None;break
            if len(records)==before:
                entry=f"year={year} page={page}: duplicate page/no pagination progress"
                failures.append(entry);year_failure_pages.append(page);next_page=page;break
            next_page=page+1
            if page==MAX_PAGES_PER_COMPANY_YEAR:
                entry=f"year={year}: reached page cap {MAX_PAGES_PER_COMPANY_YEAR}"
                failures.append(entry);year_failure_pages.append(page);next_page=page;break
        year_status[key]={"complete":year_complete,"completed_pages":sorted(completed_pages),
            "failed_pages":sorted(set(year_failure_pages)),"next_page":next_page,
            "window_start":str(lo.date()),"window_end":str(hi.date()),
            "rows_n":sum(1 for r in records.values() if str(r.get("receipt_date",""))[:4]==key),
            "requests_n_this_run":year_requests_n}
    rows=sorted(records.values(),key=lambda r:(r["receipt_date"],r.get("rcpNo") or ""))
    completed_years_n=sum(bool(year_status.get(str(year),{}).get("complete")) for year in requested_years)
    complete=completed_years_n==len(requested_years)
    record_years=sorted({int(str(row["receipt_date"])[:4]) for row in rows if str(row.get("receipt_date",""))[:4].isdigit()})
    requested_years_n=len(requested_years)
    available_years_n=sum(year in record_years for year in requested_years)
    year_coverage_pct=100.0*available_years_n/requested_years_n if requested_years_n else 0.0
    available=bool(rows) and year_coverage_pct>=80.0
    partial=bool(rows) and (not complete or available_years_n<requested_years_n)
    diag={"ticker":ticker,"company_name":name,"source":"DART public HTML detailSearch.ax",
        "requested_start":str(start.date()),"requested_end":str(end.date()),"first_date":rows[0]["receipt_date"] if rows else None,
        "last_date":rows[-1]["receipt_date"] if rows else None,"rows_n":len(rows),"requests_n":requests_n,
        "pages_n":pages_n,"attempted":requests_n>0,"available":available,"partial":partial,
        "year_coverage_pct":round(year_coverage_pct,2),"available_years_n":available_years_n,
        "requested_years_n":requested_years_n,"years_with_records":record_years,
        "query_complete":complete,"complete":complete,"year_status":year_status,
        "request_interval_seconds":request_interval_seconds(),
        "cache_hit":False,"failures":failures,"retry_log":attempt_log}
    path.write_text(json.dumps({"records":rows,"diagnostics":diag},ensure_ascii=False,indent=2),encoding="utf-8")
    if not rows and failures:
        # Keep the partial cache and diagnostics, but callers decide whether
        # the missing feature set is eligible. Do not fabricate zero counts.
        return {"records":[],"diagnostics":diag}
    return {"records":rows,"diagnostics":diag}


def disclosure_features_for_signal(records,signal_date,price_index,available=True):
    """Build disclosure features using only receipt dates strictly before signal date."""
    names=("disclosure_periodic_days_since","disclosure_bad_20d","disclosure_bad_60d",
           "disclosure_positive_20d","disclosure_count_60d")
    if not available:return {k:float("nan") for k in names}
    signal=pd.Timestamp(signal_date).normalize()
    eligible=[]
    for row in records or []:
        try:receipt=pd.Timestamp(row["receipt_date"]).normalize()
        except (KeyError,TypeError,ValueError):continue
        if receipt<signal:eligible.append((receipt,str(row.get("title", ""))))
    ix=pd.DatetimeIndex(pd.to_datetime(price_index)).tz_localize(None).normalize() if getattr(pd.DatetimeIndex(pd.to_datetime(price_index)),"tz",None) is not None else pd.DatetimeIndex(pd.to_datetime(price_index)).normalize()
    loc=int(ix.searchsorted(signal,side="left"))
    lookback_dates={window:(ix[max(0,loc-window)] if len(ix) and loc>=0 else signal-pd.Timedelta(days=window*2)) for window in (20,60)}
    recent20=[x for x in eligible if x[0]>=lookback_dates[20]]
    recent60=[x for x in eligible if x[0]>=lookback_dates[60]]
    bad20=sum(bool(bad_disclosure_keyword_matches(title)) for _,title in recent20)
    bad60=sum(bool(bad_disclosure_keyword_matches(title)) for _,title in recent60)
    positive20=sum(any(word in title for word in POSITIVE_KEYWORDS) for _,title in recent20)
    periodic=[d for d,title in eligible if any(word in title for word in PERIODIC_KEYWORDS)]
    if periodic:
        last=max(periodic);days=int(((ix>last)&(ix<=signal)).sum())
    else:days=float("nan")
    return {"disclosure_periodic_days_since":days,"disclosure_bad_20d":float(bad20),
        "disclosure_bad_60d":float(bad60),"disclosure_positive_20d":float(positive20),
        "disclosure_count_60d":float(len(recent60))}


def collect_company_disclosures(entries,*,time_budget_seconds=None,workers=None,fetcher=None,clock=None,progress=None):
    """Collect a list of (ticker, name) rows within one wall-clock budget."""
    from concurrent.futures import ThreadPoolExecutor,as_completed
    if fetcher is None:fetcher=get_company_disclosures
    import time as _time
    if clock is None:clock=_time.monotonic
    budget=float(os.environ.get("DISCLOSURE_BUDGET_SECONDS","900") if time_budget_seconds is None else time_budget_seconds)
    requested_workers=int(os.environ.get("DISCLOSURE_WORKERS","2") if workers is None else workers)
    worker_count=min(3,max(1,requested_workers))
    started=clock();results={};failed=[];partial=[];attempted=[];idx=0;completed=0
    while idx<len(entries) and max(0.0,clock()-started)<max(0.0,budget):
        batch=entries[idx:idx+worker_count];idx+=len(batch);attempted.extend(str(x[0]) for x in batch)
        with ThreadPoolExecutor(max_workers=min(worker_count,len(batch))) as pool:
            futures={pool.submit(fetcher,str(ticker),str(name),start_date="2015-12-01",end_date="2026-11-13"):(ticker,name) for ticker,name in batch}
            for future in as_completed(futures):
                completed+=1;ticker,name=futures[future]
                try:
                    result=future.result();results[str(ticker)]={**result,"ticker":str(ticker),"company_name":str(name)}
                    diag=result.get("diagnostics",{})
                    if diag.get("partial"):
                        partial.append({"ticker":str(ticker),"year_coverage_pct":diag.get("year_coverage_pct"),
                            "available":bool(diag.get("available")),"failures":diag.get("failures",[])})
                    if not diag.get("available"):
                        failed.append({"ticker":str(ticker),"error":"unavailable disclosure history",
                            "rows_n":len(result.get("records",[])),"year_coverage_pct":diag.get("year_coverage_pct"),
                            "failures":diag.get("failures",[])})
                    if progress:progress(completed,len(entries),ticker,result)
                except Exception as exc:
                    item={"ticker":str(ticker),"company_name":str(name),"error":f"{type(exc).__name__}: {str(exc)[:300]}"}
                    failed.append(item);results[str(ticker)]={"records":[],"diagnostics":{"ticker":str(ticker),"available":False,"complete":False,"failures":[item["error"]]}}
                    if progress:progress(completed,len(entries),ticker,results[str(ticker)])
        if max(0.0,clock()-started)>=budget:break
    remaining=[str(ticker) for ticker,_ in entries[idx:]]
    return {"requested_symbols":len(entries),"attempted_symbols":len(attempted),
        "available_symbols":sum(bool(x.get("diagnostics",{}).get("available")) for x in results.values()),
        "partial_symbols":partial,
        "request_interval_seconds":request_interval_seconds(),
        "budget_seconds":budget,"budget_used_seconds":round(max(0.0,clock()-started),3),
        "budget_exhausted":bool(remaining),"workers":worker_count,"results":results,
        "failures":failed,"not_attempted":remaining}
