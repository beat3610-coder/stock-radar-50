"""Free, unauthenticated DART disclosure search with resumable local cache."""
from __future__ import annotations

from datetime import date
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import re
import time
from urllib.parse import parse_qs, urlparse

import pandas as pd

DISCLOSURE_URL="https://dart.fss.or.kr/dsab007/detailSearch.ax"
DEFAULT_CACHE=Path(".cache/stock-radar/disclosures")
MAX_PAGES_PER_COMPANY_YEAR=30
MAX_RESULTS=100
BAD_KEYWORDS=("유상증자","전환사채","신주인수권","횡령","배임","감사의견","관리종목",
 "상장폐지","불성실공시","영업정지","채무불이행","소송","최대주주","공급계약 해지")
POSITIVE_KEYWORDS=("자기주식취득","자사주취득","자기주식 취득","단일판매·공급계약체결",
 "단일판매공급계약체결","무상증자")
PERIODIC_KEYWORDS=("사업보고서","반기보고서","분기보고서")


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
    if cached:
        diag=cached.get("diagnostics",{})
        if (diag.get("complete") and diag.get("requested_start","9999-12-31")<=str(start.date())
                and diag.get("requested_end","0000-01-01")>=str(end.date())):
            records=[r for r in cached["records"] if str(start.date())<=r["receipt_date"]<=str(end.date())]
            return {"records":records,"diagnostics":{**diag,"cache_hit":True,"ticker":ticker}}

    if session is None:session=requests.Session()
    try:session.headers.update({"User-Agent":"Mozilla/5.0 (compatible; stock-radar-research/1.0)","Accept":"text/html,application/xhtml+xml"})
    except Exception:pass
    records={r.get("rcpNo") or (r.get("receipt_date"),r.get("title")):r for r in (cached or {}).get("records",[])
             if r.get("receipt_date") and str(start.date())<=r["receipt_date"]<=str(end.date())}
    requests_n=0;failures=[];pages_n=0;attempt_log=[]
    for year in range(start.year,end.year+1):
        lo=max(start,pd.Timestamp(year=year,month=1,day=1));hi=min(end,pd.Timestamp(year=year,month=12,day=31))
        for page in range(1,MAX_PAGES_PER_COMPANY_YEAR+1):
            params={"currentPage":page,"maxResults":MAX_RESULTS,"sort":"date","series":0,
                "textCrpNm":name,"startDate":lo.strftime("%Y%m%d"),"endDate":hi.strftime("%Y%m%d")}
            html=None
            for attempt in range(1,4):
                requests_n+=1
                try:
                    response=session.get(DISCLOSURE_URL,params=params,timeout=20)
                    if response.status_code in (429,500,502,503,504):
                        raise requests.HTTPError(f"HTTP {response.status_code}",response=response)
                    response.raise_for_status();html=response.text;break
                except Exception as exc:
                    entry=f"year={year} page={page} try={attempt}: {type(exc).__name__}: {str(exc)[:180]}"
                    attempt_log.append(entry)
                    if attempt<3:time.sleep(.4*(2**(attempt-1)))
                    else:failures.append(entry)
            if html is None:break
            page_rows=parse_disclosure_html(html);pages_n+=1
            if not page_rows:break
            before=len(records)
            for row in page_rows:
                if row["rcpNo"]:key=row["rcpNo"]
                else:key=(row["receipt_date"],row["title"],row.get("submitter"))
                records[key]=row
            if len(page_rows)<MAX_RESULTS:break
            if len(records)==before:
                failures.append(f"year={year} page={page}: duplicate page/no pagination progress")
                break
            if page==MAX_PAGES_PER_COMPANY_YEAR:
                failures.append(f"year={year}: reached page cap {MAX_PAGES_PER_COMPANY_YEAR}")
    rows=sorted(records.values(),key=lambda r:(r["receipt_date"],r.get("rcpNo") or ""))
    complete=not failures
    diag={"ticker":ticker,"company_name":name,"source":"DART public HTML detailSearch.ax",
        "requested_start":str(start.date()),"requested_end":str(end.date()),"first_date":rows[0]["receipt_date"] if rows else None,
        "last_date":rows[-1]["receipt_date"] if rows else None,"rows_n":len(rows),"requests_n":requests_n,
        "pages_n":pages_n,"attempted":requests_n>0,"available":bool(rows) and complete,
        "query_complete":complete,"complete":complete,
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
    bad20=sum(any(word in title for word in BAD_KEYWORDS) for _,title in recent20)
    bad60=sum(any(word in title for word in BAD_KEYWORDS) for _,title in recent60)
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
    worker_count=max(1,int(os.environ.get("DISCLOSURE_WORKERS","6") if workers is None else workers))
    started=clock();results={};failed=[];attempted=[];idx=0;completed=0
    while idx<len(entries) and max(0.0,clock()-started)<max(0.0,budget):
        batch=entries[idx:idx+worker_count];idx+=len(batch);attempted.extend(str(x[0]) for x in batch)
        with ThreadPoolExecutor(max_workers=min(worker_count,len(batch))) as pool:
            futures={pool.submit(fetcher,str(ticker),str(name),start_date="2015-12-01",end_date="2026-11-13"):(ticker,name) for ticker,name in batch}
            for future in as_completed(futures):
                completed+=1;ticker,name=futures[future]
                try:
                    result=future.result();results[str(ticker)]={**result,"ticker":str(ticker),"company_name":str(name)}
                    if progress:progress(completed,len(entries),ticker,result)
                except Exception as exc:
                    item={"ticker":str(ticker),"company_name":str(name),"error":f"{type(exc).__name__}: {str(exc)[:300]}"}
                    failed.append(item);results[str(ticker)]={"records":[],"diagnostics":{"ticker":str(ticker),"available":False,"complete":False,"failures":[item["error"]]}}
                    if progress:progress(completed,len(entries),ticker,results[str(ticker)])
        if max(0.0,clock()-started)>=budget:break
    remaining=[str(ticker) for ticker,_ in entries[idx:]]
    return {"requested_symbols":len(entries),"attempted_symbols":len(attempted),
        "available_symbols":sum(bool(x.get("diagnostics",{}).get("available")) for x in results.values()),
        "budget_seconds":budget,"budget_used_seconds":round(max(0.0,clock()-started),3),
        "budget_exhausted":bool(remaining),"workers":worker_count,"results":results,
        "failures":failed,"not_attempted":remaining}
