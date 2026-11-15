"""300-stock online directional audit. Research-only, no strategy approval.

UP/DOWN/SIDEWAYS fixed 10-session close-to-close horizon and +/-3% band.
Prequential evaluations never access a future label before it matures.
"""
from datetime import datetime,timezone
import json
from engine import compute
from providers import get_prices
from universe import get_universe
from direction3 import make_direction_rows,score_directions,HORIZON,BAND_PCT
from buy_success import attach_buy_labels,score_buy_success

def run_direction3(per_market=100):
    univ=get_universe()
    result={"created_utc":datetime.now(timezone.utc).isoformat(),
            "type":"direction3_10session_3pct",
            "target_precision_pct_each":80,
            "fixed_horizon_sessions":HORIZON,"sideways_band_pct":BAND_PCT,
            "research_only":True,"live_approved":False,
            "evaluation_warning":"Retrospective period already reviewed; not independent prospective evidence",
            "markets":{}}
    for market in ("KR","KQ","US"):
        excluded={"KR":{"472150","0262B0"},"KQ":set(),"US":{"SOXL"}}[market]
        symbols=[x[0] for x in univ[market] if x[0] not in excluded][:per_market]
        rows=[];buy_rows=[];loaded=0;errors=[]
        for i,ticker in enumerate(symbols,1):
            try:
                data,_=get_prices(ticker,market)
                x=compute(data)
                direction_rows=make_direction_rows(x,ticker,market,end_date="2026-11-01")
                rows.extend(direction_rows)
                buy_rows.extend(attach_buy_labels(x,direction_rows))
                loaded+=1
            except Exception as exc:
                errors.append({"ticker":ticker,"error":str(exc)[:160]})
            if i%20==0:
                print(f"DIRECTION_PROGRESS {market} {i}/{len(symbols)} loaded={loaded} labels={len(rows)}",flush=True)
        metrics=score_directions(rows)
        buy_metrics=score_buy_success(buy_rows)
        result["markets"][market]={"loaded":loaded,"requested":len(symbols),"errors":errors[:10],
                                  "labels_total":len(rows),"buy_labels_total":len(buy_rows),
                                  "buy_success":buy_metrics,**metrics}
        slim=metrics["windows"]["diagnostic"]
        print("DIRECTION_MARKET "+market+" "+json.dumps({
            "loaded":loaded,"labels_total":len(rows),
            "buy_labels_total":len(buy_rows),
            "buy_diagnostic":buy_metrics["windows"]["diagnostic"],
            "diagnostic":slim,
            "validation":metrics["windows"]["validation"],
            "trained_outcomes":metrics["trained_outcomes"]},ensure_ascii=False),flush=True)
    return result

if __name__=="__main__":
    print("DIRECTION300_RESULT "+json.dumps(run_direction3(),ensure_ascii=False),flush=True)
