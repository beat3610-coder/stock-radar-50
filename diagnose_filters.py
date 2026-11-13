"""Paired, fixed-calendar diagnosis of old-vs-risk-screen entries.
Same symbols, dates, signals, next-open fills, and roundtrip costs.
Research diagnostic only; no claim of independent predictive validation.
"""
import json
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
from audit_300 import simulate_exit, metrics
from engine import compute,evaluate
from providers import get_prices
from universe import get_universe

CUTOFF = "2026-04-01"  # fixed IN ADVANCE of comparing variants
LAST_SIGNAL = "2026-11-01"
WARMUP = 270
STEP = 5
OLD_BUY = {"매수 검토","강한 매수 검토","기술매수 후보·수급 미확인"}
NEW_BUY = {"연구용 매수 후보·실거래 미승인", "연구용 후보·위험조건 미충족"}
HORIZON=10

def one_market(items,market):
    old=[];new=[];counts={"dates_reviewed":0,"old_candidates":0,
                          "risk_passes":0,"risk_rejected":0,
                          "bad_price_plan":0,"gap_skip":0}
    reasons={}
    errors=[]
    for number,(ticker,name,_,sector) in enumerate(items,1):
        try:
            raw,provider=get_prices(ticker,market)
            x=compute(raw)
            candidates=[]
            for t in range(WARMUP,len(x)-HORIZON,STEP):
                date=str(pd.Timestamp(x.index[t]).date())
                if date>LAST_SIGNAL: break
                info=evaluate(x.iloc[:t+1],market=market,flow=None)
                counts["dates_reviewed"]+=1
                if info.get("signal_before_risk") not in OLD_BUY:continue
                counts["old_candidates"]+=1
                if info["signal"] in NEW_BUY:counts["risk_passes"]+=1
                else:
                    counts["risk_rejected"]+=1
                    reason=info.get("entry_reason","unknown")
                    # Each reason might contain pre-risk notes; inspect high-level.
                    for k in ("상승 국면","횡보","하락","기술점수","RSI",
                              "변동성","손절폭","손익비","과거 평균 손익"):
                        if k in reason: reasons[k]=reasons.get(k,0)+1
                op=float(x.Open.iloc[t+1]);close=float(info["price"])
                stop=info.get("stop");take=info.get("trade_target")
                if stop is None or take is None or op<=0:
                    counts["bad_price_plan"]+=1;continue
                if abs(op/close-1)>0.025:
                    counts["gap_skip"]+=1;continue
                risk=(op-float(stop))/op
                reward=(float(take)-op)/op
                if not(0<risk<=0.12 and reward>=0.02 and reward/risk>=1.2):
                    counts["bad_price_plan"]+=1;continue
                trade=simulate_exit(x,t,op,float(stop),float(take))
                if trade:
                    candidates.append({**trade,"ticker":ticker,
                         "entry_date":str(pd.Timestamp(x.index[t+1]).date()),
                         "signal_date":date,"risk_pass":info["signal"] in NEW_BUY})
            # Apply non-overlap consistently for both variants. Both start from
            # exactly the same potential entries and same stop/take plans.
            for variant,output in ((False,old),(True,new)):
                blocked=-1
                for item in sorted(candidates,key=lambda z:z["signal_date"]):
                    bar=x.index.get_loc(pd.Timestamp(item["signal_date"]))
                    if bar<=blocked or (variant and not item["risk_pass"]):continue
                    blocked=item["exit_index"]
                    output.append({k:v for k,v in item.items() if k!="risk_pass"})
            if number%5==0:
                print(f"COMPARE_PROGRESS {market} {number}/{len(items)} old={len(old)} new={len(new)}",flush=True)
        except Exception as exc:
            errors.append({"ticker":ticker,"reason":str(exc)[:200]})
    def section(trades,side):
        chosen=[t for t in trades if (t["entry_date"]<CUTOFF if side=="development"
                                      else t["entry_date"]>=CUTOFF)]
        return metrics(chosen)
    return {"requested":len(items),"errors":errors,"counts":counts,
            "rejection_keywords":reasons,
            "baseline":{"development":section(old,"development"),
                        "holdout":section(old,"holdout")},
            "risk_screen":{"development":section(new,"development"),
                           "holdout":section(new,"holdout")}}

def run(per_market=20):
    u=get_universe()
    result={"as_of_utc":datetime.now(timezone.utc).isoformat(),
            "calendar_cutoff":CUTOFF,"signal_end":LAST_SIGNAL,
            "scan_every_n_trading_sessions":STEP,
            "limitations":["same-day OHLC stop-first conservative",
                           "selected-current-constituents survivorship bias",
                           "no portfolio synchronization",
                           "not independent of prior strategy decisions",
                           "baseline uses current historical scenario algorithm",
                           "baseline is signal_before_risk not fully reverted old engine"],
            "markets":{}}
    for market in ("KR","KQ","US"):
        skip={"KR":{"472150","0262B0"},"KQ":set(),"US":{"SOXL"}}[market]
        items=[x for x in u[market] if x[0] not in skip][:per_market]
        result["markets"][market]=one_market(items,market)
    return result

if __name__=="__main__":
    x=run()
    Path("data").mkdir(exist_ok=True)
    Path("data/paired-diagnosis-60.json").write_text(json.dumps(x,ensure_ascii=False,indent=2,default=str),encoding="utf-8")
    print("COMPARE_RESULT "+json.dumps(x,ensure_ascii=False,default=str),flush=True)
