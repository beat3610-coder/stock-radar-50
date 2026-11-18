"""Strict Naver market-cap constituent fallback; research-only snapshot.

Public endpoint may change without notice. Fail closed on unexpected payloads.
Never silently substitute an ALL-market list for a KR/KQ ranking.
"""
import re
from datetime import date

ENDPOINT="https://stock.naver.com/api/domestic/market/stock/default"

def naver_api_top100(market,session=None):
    if market not in ("KR","KQ"):
        raise ValueError("KR/KQ only")
    import requests
    session=session or requests.Session()
    session.headers.update({"User-Agent":"Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/131.0 Safari/537.36",
        "Accept":"application/json, text/plain, */*", "Referer":"https://stock.naver.com/"})
    target="KOSPI" if market=="KR" else "KOSDAQ"
    rows=[];seen=set();problems=[]
    for page in range(5):
        resp=session.get(ENDPOINT,params={"tradeType":"KRX","marketType":target,
                         "orderType":"marketSum","startIdx":page,"pageSize":100},timeout=12)
        resp.raise_for_status()
        data=resp.json()
        if not isinstance(data,(list,dict)):
            raise ValueError("unexpected top-level JSON")
        items=data if isinstance(data,list) else data.get("items")
        if items is None and isinstance(data,dict):
            for container in ("data","result"):
                value=data.get(container)
                if isinstance(value,list):items=value;break
                if isinstance(value,dict) and isinstance(value.get("items"),list):
                    items=value["items"];break
        if not isinstance(items,list):
            raise ValueError("market-cap list absent from API JSON")
        if not items:break
        for item in items:
            if not isinstance(item,dict):continue
            code=str(item.get("itemcode") or item.get("itemCode") or item.get("code") or item.get("stockCode") or "").strip()
            name=str(item.get("itemname") or item.get("stockName") or item.get("itemName") or item.get("name") or "").strip()
            classification=str(item.get("marketType") or item.get("market") or "").upper()
            sosok=item.get("sosok")
            if sosok is not None:
                if str(sosok)!=("0" if market=="KR" else "1"):
                    problems.append("sosok mismatch")
                    continue
            elif classification and target not in classification and classification not in (("STK",) if market=="KR" else ("KSQ",)):
                problems.append("mixed-market item")
                continue
            elif not classification:
                problems.append("missing market classifier")
                continue
            if (not re.fullmatch(r"[0-9]{6}",code) or not name or code in seen or
                re.search(r"스팩|SPAC|ETF|ETN|우선주|우$|우B$|우C$|우[0-9]",name,re.I)):
                continue
            rows.append((code,name,"업종 확인 불가"));seen.add(code)
        if len(rows)>=100:break
        if len(items)<100:break
    if len(rows)<100:
        raise RuntimeError(f"Naver stock market API {target} incomplete {len(rows)}/100 ({';'.join(problems[:2])})")
    return rows[:100],f"Naver stock API {target} marketSum current ranking / {date.today()} / research-only snapshot (not PIT)"
