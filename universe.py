"""STOCK RADAR: 100 largest KOSPI ordinary shares + 100 Nasdaq companies, with 3 fixed ETFs.
Nasdaq cached snapshot curated from index publications in November 2026.
Constituent feeds can fail: caller must show source/fallback and never silently label old data current.
"""
import pandas as pd
import datetime as dt

MANDATORY_KR=[("472150","TIGER 배당커버드콜액티브","필수 ETF"),
              ("0262B0","TIGER 200커버드콜액티브","필수 ETF")]
MANDATORY_US=[("SOXL","SOXL","필수 3배 레버리지 ETF")]
NASDAQ_SNAPSHOT="ADBE AMD ABNB ALNY GOOGL AMZN AEP AMGN ADI AAPL AMAT APP ARM ASML ALAB ADSK ADP AXON BKR BKNG AVGO CDNS CTAS CSCO CCEP CMCSA CEG CPRT CRWV COST CRWD CSX DDOG DXCM FANG DASH EXC FAST FER FTNT GEHC GILD HONA HON IDXX INTC INTU ISRG KDP KLAC LRCX LIN LITE MAR MRVL MELI META MCHP MU MSFT MSTR MDLZ MPWR MNST NBIS NFLX NVDA NXPI ORLY ODFL PCAR PLTR PANW PAYX PYPL PDD PEP QCOM REGN RKLB ROP ROST SNDK STX SHOP SPCX SBUX SNPS TMUS TTWO TER TSLA TXN TRI VRTX WMT WDC WDAY XEL MRNA".split()
assert len(NASDAQ_SNAPSHOT)==100 and len(set(NASDAQ_SNAPSHOT))==100
SNAPSHOT_DATE="2026-11-13"
# Keep no stale old watchlist shortcuts - all included stocks have a provenance.
def kospi_top100():
    import FinanceDataReader as fdr
    attempts=[]
    for market in ("KOSPI","KRX"):
        try:
            df=fdr.StockListing(market)
            df.columns=[str(c).strip() for c in df.columns]
            cap=next((c for c in df if c.lower() in ("marcap","marketcap","market_cap","marketcapitalization")),None)
            ticker=next((c for c in df if c.lower() in ("code","symbol","ticker")),None)
            name=next((c for c in df if c.lower() in ("name","company")),None)
            if not all((cap,ticker,name)):raise ValueError(f"열 누락: {list(df.columns)}")
            if market=="KRX":
                market_col=next((c for c in df if c.lower() in ("market","marketid")),None)
                if not market_col:raise ValueError("KRX 시장구분 열 없음")
                df=df[df[market_col].astype(str).str.upper().isin(("KOSPI","STK"))]
            df[cap]=pd.to_numeric(df[cap],errors="coerce")
            df[ticker]=df[ticker].astype(str).str.strip().str.zfill(6)
            # Ordinary company stocks only; remove preferred shares (last 1/5/7/9), ETFs and SPACs.
            df=df[df[ticker].str.fullmatch(r"\\d{6}") & df[cap].gt(0)]
            df=df[~df[name].astype(str).str.contains("스팩|SPAC|ETF|ETN|우선주",case=False,na=False)]
            df=df[~df[ticker].str[-1].isin(list("1579"))]
            df=df.sort_values(cap,ascending=False).drop_duplicates(ticker)
            rows=[(str(row[ticker]),str(row[name]),"코스피 시총 상위") for _,row in df.head(100).iterrows()]
            if len(rows)!=100:raise ValueError(f"코스피 종목 부족: {len(rows)}/100")
            return rows,f"FinanceDataReader {market} 시총순 정렬 / 조회일 {dt.date.today().isoformat()}"
        except Exception as e:attempts.append(f"{market}: {str(e)[:170]}")
    raise RuntimeError("코스피 100종목 구성을 가져오지 못했습니다. "+" | ".join(attempts))

def nasdaq_top100():
    # Use independently dated index list rather than top 100 Nasdaq exchange shares.
    try:
        import requests
        from io import StringIO
        url="https://en.wikipedia.org/wiki/List_of_NASDAQ-100_companies"
        resp=requests.get(url,headers={"User-Agent":"StockRadar50/1.0 (public educational tool)"},timeout=12)
        resp.raise_for_status()
        tables=pd.read_html(StringIO(resp.text))
        candidates=[]
        for t in tables:
            t.columns=[str(x).strip().lower() for x in t.columns]
            col=next((x for x in t.columns if "ticker" in x),None)
            if col and len(t)>=95:
                cols=list(t.columns)
                company=next((x for x in cols if "company" in x),None)
                sector=next((x for x in cols if "industry" in x),None)
                for _,r in t.iterrows():
                    code=str(r[col]).strip().upper().replace(".","-")
                    if code=="GOOG":continue  # Same underlying Alphabet company as GOOGL
                    if code.isalnum() and len(code)<=6:
                        candidates.append((code,str(r[company]) if company else code,str(r[sector]) if sector else "Nasdaq-100"))
                if len(candidates)==100 and len(set(x[0] for x in candidates))==100:
                    return candidates,"Wikipedia Nasdaq-100 구성 목록 / 실시간 조회"
        raise ValueError(f"공개 표의 100기업 수 검증 실패: {len(candidates)}")
    except Exception as e:
        rows=[(code,code,"Nasdaq-100") for code in NASDAQ_SNAPSHOT]
        return rows,f"2026-11-13 기준 Nasdaq-100 저장 목록 (구성 자동갱신 실패: {str(e)[:90]})"

def get_universe():
    kr_error=None
    try: kr,kr_source=kospi_top100()
    except Exception as e: kr=[];kr_source="시총 상위 100종목 조회 실패";kr_error=str(e)
    us,us_source=nasdaq_top100()
    kr_codes={code for code,_,_ in kr}
    us_codes={code for code,_,_ in us}
    # In the event a mandatory instrument qualifies for the index, do not duplicate.
    kr.extend([x for x in MANDATORY_KR if x[0] not in kr_codes])
    us.extend([x for x in MANDATORY_US if x[0] not in us_codes])
    return {
      "KR":[(c,n,"KR",s) for c,n,s in kr],
      "US":[(c,n,"US",s) for c,n,s in us],
      "KR_source":kr_source,"US_source":us_source,"KR_error":kr_error,
      "KR_index_count":len(kr_codes),"US_index_count":len(us_codes)
    }
