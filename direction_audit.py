"""300-stock continuing historical formula research. Research-only.

Every run creates a new formula generation around the previous winner and ranks
it across all eligible historical half-year slices. Predictions remain causal:
an outcome is learned only after its label-availability date.
"""
from datetime import datetime,timezone
from pathlib import Path
import json
import os
from engine import compute
from providers import get_prices
from universe import get_universe
from direction3 import make_direction_rows,score_directions,HORIZON,BAND_PCT
from buy_success import attach_buy_labels,score_buy_success
from adaptive_trade_plans import label_plans,score_plans


def run_direction3(per_market=100,previous=None):
    previous=previous or {}
    generation=int(previous.get("learning_generation",0) or 0)+1
    univ=get_universe()
    cutoff=datetime.now(timezone.utc).date().isoformat()
    result={"created_utc":datetime.now(timezone.utc).isoformat(),
            "type":"direction3_10session_3pct",
            "target_precision_pct_each":80,
            "fixed_horizon_sessions":HORIZON,"sideways_band_pct":BAND_PCT,
            "evaluation_as_of":cutoff,
            "learning_generation":generation,
            "previous_report_created_utc":previous.get("created_utc"),
            "search_scope":"전체 완료 과거기간 반기별 순차평가 + 이전 최적식 주변 신규 변이",
            "research_only":True,"live_approved":False,
            "independent_holdout":False,
            "evaluation_warning":"전체 과거 구간을 수식 선택에 사용한 연구 적합도이며 미래 성공률 보장이 아닙니다.",
            "markets":{}}
    selected=os.environ.get("STOCK_AUDIT_MARKET","").strip().upper()
    markets=(selected,) if selected in ("KR","KQ","US") else ("KR","KQ","US")
    previous_markets=previous.get("markets") or {}
    for market in markets:
        excluded={"KR":{"472150","0262B0"},"KQ":set(),"US":{"SOXL"}}[market]
        symbols=[x[0] for x in univ[market] if x[0] not in excluded][:per_market]
        rows=[];buy_rows=[];plan_rows=[];loaded=0;errors=[]
        for i,ticker in enumerate(symbols,1):
            try:
                data,_=get_prices(ticker,market)
                x=compute(data)
                direction_rows=make_direction_rows(x,ticker,market,end_date=cutoff)
                rows.extend(direction_rows)
                buy_rows.extend(attach_buy_labels(x,direction_rows))
                plan_rows.extend(label_plans(x,direction_rows))
                loaded+=1
            except Exception as exc:
                errors.append({"ticker":ticker,"error":str(exc)[:160]})
            if i%20==0:
                print(f"DIRECTION_PROGRESS {market} {i}/{len(symbols)} loaded={loaded} labels={len(rows)}",flush=True)
        previous_market=previous_markets.get(market) or {}
        previous_direction=((previous_market.get("tuning") or {}).get("chosen") or None)
        previous_buy=((((previous_market.get("buy_success") or {})
                        .get("tuning") or {}).get("chosen")) or None)
        metrics=score_directions(rows,eval_end=cutoff,
            search_generation=generation,previous_best=previous_direction)
        buy_metrics=score_buy_success(buy_rows,eval_end=cutoff,
            search_generation=generation,previous_best=previous_buy)
        adaptive_plans=score_plans(plan_rows,eval_end=cutoff)
        result["markets"][market]={"loaded":loaded,"requested":len(symbols),
                                  "errors":errors[:10],
                                  "labels_total":len(rows),
                                  "buy_labels_total":len(buy_rows),
                                  "buy_success":buy_metrics,
                                  "adaptive_plans":adaptive_plans,**metrics}
        slim=metrics["windows"]["diagnostic"]
        print("DIRECTION_MARKET "+market+" "+json.dumps({
            "generation":generation,"loaded":loaded,"labels_total":len(rows),
            "buy_labels_total":len(buy_rows),
            "buy_diagnostic":buy_metrics["windows"]["diagnostic"],
            "adaptive_plan_diagnostic":adaptive_plans["windows"]["diagnostic"],
            "diagnostic":slim,
            "whole_history_direction":metrics["tuning"],
            "whole_history_buy":buy_metrics["tuning"],
            "trained_outcomes":metrics["trained_outcomes"]},
            ensure_ascii=False),flush=True)
    return result


if __name__=="__main__":
    report_path=Path("data/direction3-rolling-latest.json")
    previous={}
    if report_path.exists():
        try:
            previous=json.loads(report_path.read_text(encoding="utf-8"))
        except Exception:
            previous={}
    report=run_direction3(previous=previous)
    report_path.parent.mkdir(exist_ok=True)
    report_path.write_text(json.dumps(report,ensure_ascii=False,indent=2),
                           encoding="utf-8")
    print("DIRECTION300_RESULT "+json.dumps(report,ensure_ascii=False),flush=True)
