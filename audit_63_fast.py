"""Fast, transparent 63-instrument historical audit (research only).
30 KOSPI + 30 KOSDAQ + 2 TIGER ETFs + SOXL. No Nasdaq ordinary shares.
Select strategy on development period only; report fixed validation outcomes
even if win rate is low. Per-instrument n=0 is not a 0% win rate.
"""
import json
from pathlib import Path
from datetime import datetime, timezone
from universe import get_universe
from providers import get_prices
from strategy_lab import candidates_from_frame, TRAIN_END, VALID_END, HOLDOUT_END
from audit_300 import metrics

def run():
    universe=get_universe(include_us=False)
    cohorts={
        "KOSPI":universe["KR"][:30],
        "KOSDAQ":universe["KQ"][:30],
        "TIGER_ETF":universe["KR"][-2:],
        "SOXL":universe["US"][-1:],
    }
    out={"created_utc":datetime.now(timezone.utc).isoformat(),
         "status":"RESEARCH_NOT_LIVE_APPROVED",
         "selection":"development-period highest mean net return, minimum 10 trades; validation never used for selection",
         "windows":{"development_end":TRAIN_END,"validation_end":VALID_END,"diagnostic_end":HOLDOUT_END},
         "cohorts":{}}
    for cohort,items in cohorts.items():
        rows=[];pool={};errors=[]
        for i,(ticker,name,market,sector) in enumerate(items,1):
            try:
                prices,source=get_prices(ticker,market)
                _,variants=candidates_from_frame(prices,ticker,market)
                for strategy,trades in variants.items():
                    pool.setdefault(strategy,[]).extend(trades)
                rows.append({"ticker":ticker,"name":name,"market":market,"source":str(source),
                             "variants":variants})
            except Exception as exc:
                errors.append({"ticker":ticker,"name":name,"error":str(exc)[:180]})
            print(f"AUDIT_PROGRESS {cohort} {i}/{len(items)} loaded={len(rows)}",flush=True)
        training={}
        for strategy,trades in pool.items():
            dev=[t for t in trades if t["entry_date"]<TRAIN_END and t["exit_date"]<TRAIN_END]
            training[strategy]=metrics(dev)
        eligible=[s for s,m in training.items() if m["trades"]>=10 and m["mean_net_pct"] is not None]
        selected=max(eligible,key=lambda s:training[s]["mean_net_pct"],default=None)
        val=[];diag=[];instruments=[]
        for row in rows:
            ts=row["variants"].get(selected,[]) if selected else []
            v=[t for t in ts if TRAIN_END<=t["entry_date"] and t["exit_date"]<VALID_END]
            d=[t for t in ts if VALID_END<=t["entry_date"] and t["exit_date"]<HOLDOUT_END]
            val.extend(v);diag.extend(d)
            instruments.append({"ticker":row["ticker"],"name":row["name"],"market":row["market"],
                "validation":metrics(v),"diagnostic":metrics(d)})
        out["cohorts"][cohort]={"requested":len(items),"loaded":len(rows),"errors":errors,
            "selected_development_strategy":selected,
            "development":training.get(selected),"validation":metrics(val),
            "diagnostic":metrics(diag),"instruments":instruments,
            "complete":len(rows)==len(items)}
        print("AUDIT_RESULT "+json.dumps({"cohort":cohort,"requested":len(items),
            "loaded":len(rows),"strategy":selected,"validation":metrics(val),
            "diagnostic":metrics(diag)},ensure_ascii=False),flush=True)
    Path("data").mkdir(exist_ok=True)
    Path("data/audit-63-latest.json").write_text(json.dumps(out,ensure_ascii=False,indent=2,default=str),encoding="utf-8")
    return out

if __name__=="__main__":
    run()
