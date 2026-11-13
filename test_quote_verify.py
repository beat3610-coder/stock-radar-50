"""Deterministic quote cross-check unit cases; no remote calls."""
import json
import sys
import types
from pathlib import Path
from unittest.mock import patch
import pandas as pd
from providers import verify_recent_close

dates=pd.bdate_range("2026-11-09",periods=5)
def frame(close):
    return pd.DataFrame({"Open":[100]*5,"High":[102]*5,"Low":[98]*5,
                         "Close":[100]*4+[close],"Volume":[10000]*5},index=dates)
def check(data, source="FinanceDataReader NAVER:005930",date="2026-11-13"):
    with patch.dict(sys.modules,{"yfinance":types.SimpleNamespace(download=lambda *a,**kw: data)}):
        return verify_recent_close("005930","KR",date,100.0,source)
cases={}
cases["equal"]=check(frame(100))
assert cases["equal"]["status"]=="agree"
cases["mismatch"]=check(frame(102))
assert cases["mismatch"]["status"]=="mismatch" and cases["mismatch"]["difference_pct"]>0.5
cases["date_mismatch"]=check(frame(100).iloc[:-1])
assert cases["date_mismatch"]["status"]=="unverified"
cases["same_source"]=check(frame(100),source="Yahoo 005930.KS")
assert cases["same_source"]["status"]=="unverified"
cases["invalid_data"]=check(pd.DataFrame())
assert cases["invalid_data"]["status"]=="unverified"
Path("data").mkdir(exist_ok=True)
Path("data/quote-test-result.json").write_text(
    json.dumps({"success":True,"statuses":{k:v["status"] for k,v in cases.items()}},indent=2),
    encoding="utf8")
print("quote date and discrepancy guards passed:",{k:v["status"] for k,v in cases.items()})
