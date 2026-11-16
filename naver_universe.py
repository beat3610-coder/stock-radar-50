"""Secondary Korean constituent discovery via Naver Finance market-cap pages.

Used only after FinanceDataReader fails. This is a latest *current* snapshot,
NOT a historical point-in-time constituent universe and NOT licensed data for resale.
Fail closed when fewer than 100 ordinary shares can be verified.
"""
import re
from io import StringIO
from datetime import datetime
import pandas as pd

def naver_top100(market,session=None):
    if market not in ("KR","KQ"):
        raise ValueError("Unsupported market")
    import requests
    from bs4 import BeautifulSoup
    s=session or requests.Session()
    s.headers.update({"User-Agent":"Mozilla/5.0 (compatible; StockRadarResearch/1.0)"})
    market_id="0" if market=="KR" else "1"
    candidates=[]
    seen=set()
    problems=[]
    for page in range(1,7):
        url="https://finance.naver.com/sise/sise_market_sum.naver"
        try:
            resp=s.get(url,params={"sosok":market_id,"page":page},timeout=12)
            resp.raise_for_status()
            resp.encoding=resp.apparent_encoding or "euc-kr"
            soup=BeautifulSoup(resp.text,"html.parser")
            table=soup.select_one("table.type_2")
            if table is None:
                raise ValueError("market-cap table absent")
            count_before=len(candidates)
            for tr in table.select("tr"):
                anchor=tr.select_one('a[href*="main.naver?code="]')
                if anchor is None:
                    continue
                href=anchor.get("href","")
                match=re.search(r"(?:\\?|&)code=(\\d{6})(?:&|$)",href)
                if not match:
                    continue
                code=match.group(1)
                if code in seen:
                    continue
                name=anchor.get_text(" ",strip=True)
                if not name or re.search(r"스팩|SPAC|ETF|ETN|우선주|우$|우B$|우C$|우\\d",name,re.I):
                    continue
                values=[td.get_text(" ",strip=True).replace(",","") for td in tr.select("td")]
                # Rank comes from Naver market-cap page, never infer market cap
                # from unrelated cells or re-sort by an unverified column.
                if len(values)<5:
                    continue
                candidates.append((code,name,"업종 확인 불가"))
                seen.add(code)
            if len(candidates)==count_before:
                problems.append(f"page {page} empty")
        except Exception as exc:
            problems.append(f"page {page}: {str(exc)[:90]}")
        if len(candidates)>=100:
            break
    if len(candidates)<100:
        raise RuntimeError(f"Naver constituents incomplete {len(candidates)}/100; {'; '.join(problems[:3])}")
    return candidates[:100],f"Naver Finance {market} 현재 시총순 / {datetime.now().date().isoformat()} / 업종 미검증 / 과거 구성종목 아님"
