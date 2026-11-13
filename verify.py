"""Public daily vendor-data audit; no fabricated prices or signals."""
import concurrent.futures as cf
import csv
import datetime as dt
import json
import pathlib
from universe import WATCHLIST
from engine import compute,evaluate
from providers import get_prices

ROOT=pathlib.Path(__file__).resolve().parent
OUT=ROOT/"data"
OUT.mkdir(exist_ok=True)

def audit_one(entry):
    code,name,market,sector=entry
    result={"code":code,"name":name,"market":market,"sector":sector}
    try:
        bars,source=get_prices(code,market)
        r=evaluate(compute(bars))
        date=dt.date.fromisoformat(r["date"])
        age=(dt.datetime.now(dt.timezone.utc).date()-date).days
        # The 5-calendar-day policy tolerates weekends/market holidays but flags stale sources.
        status="정상" if age<=5 else "기준일 오래됨"
        result.update(status=status,source=source,date=r["date"],age_days=age,
          price=round(r["price"],3),score=r["score"],signal=r["signal"],
          rsi=round(r["rsi"],2) if r["rsi"] is not None else None,
          rsi_delta5=round(r["rsi_delta5"],2) if r["rsi_delta5"] is not None else None,
          bar_count=len(bars),error="")
    except Exception as e:
        result.update(status="조회 실패",source=None,date=None,age_days=None,
          price=None,score=None,signal=None,rsi=None,rsi_delta5=None,
          bar_count=0,error=str(e)[:350])
    print(f"[{result['status']}] {market}/{code} {name}: {result.get('date')} {result.get('source')} {result['error'][:120]}",flush=True)
    return result

def main():
    with cf.ThreadPoolExecutor(max_workers=4) as executor:
        results=list(executor.map(audit_one,WATCHLIST))
    counts={status:sum(x["status"]==status for x in results)
        for status in ("정상","기준일 오래됨","조회 실패")}
    report={"checked_at_utc":dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
      "environment":"GitHub Actions (Render runtime access may differ)",
      "count":len(results),"counts":counts,"rows":results}
    (OUT/"verification.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    with (OUT/"verification.csv").open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(results[0]));w.writeheader();w.writerows(results)
    print(f"VERIFICATION SUMMARY {counts}",flush=True)
if __name__=="__main__":main()
