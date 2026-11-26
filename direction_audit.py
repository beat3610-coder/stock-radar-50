"""300-stock continuing historical formula research. Research-only.

Each run evolves direction, probability, and executable stop/take/hold formulas
around the previous winners and ranks them across historical half-year slices.
"""
from datetime import datetime,timezone
from pathlib import Path
import json
import os
from engine import compute
from providers import get_prices,get_historical_prices
from universe import get_universe
from research_integrity import certify_all, market_health
from regime_direction import score_regime_direction
from direction3 import make_direction_rows,score_directions,HORIZON,BAND_PCT
from buy_success import attach_buy_labels,score_buy_success
from adaptive_trade_plans import (
    label_plans,score_plans,make_plan_candidates,TARGET_WIN_PCT)
from bde_dashboard import load_summary as load_bde_summary


def bde_evolution_block():
    """시간별 진화 보고서에 함께 실을 B+D+E 최근 필터 요약.

    값은 동결된 63종목 감사의 B+D+E 결과를 그대로 옮긴 것이고, 이 실행의 새 학습과
    섞지 않는다. 요약 파일이 없으면 빈 값을 넣지 않고 아예 넣지 않는다.
    """
    summary=load_bde_summary()
    if not summary:return None
    source=summary.get("source") or {}
    return {"source_report_created_utc":source.get("report_created_utc"),
            "summary_generated_utc":summary.get("generated_utc"),
            "audit_report":source.get("audit_report"),"run_id":source.get("run_id"),
            "head":source.get("head"),"stage_order":summary.get("stage_order"),
            "selected_policy":{cohort:(block.get("selected_policy") or {}).get("label")
                                for cohort,block in (summary.get("cohorts") or {}).items()},
            "recent_signal":{cohort:{"latest_signal_date":(block.get("recent_signal") or {}).get("latest_signal_date"),
                                     "accepted_n_by_stage":(block.get("recent_signal") or {}).get("accepted_n_by_stage")}
                              for cohort,block in (summary.get("cohorts") or {}).items()},
            "direction3_accuracy_pct":{market:{period:(row or {}).get("accuracy_pct")
                                               for period,row in block.items()}
                                       for market,block in (summary.get("direction3") or {}).items()},
            "notes":summary.get("notes"),"research_only":True,"live_approved":False}


def run_direction3(per_market=100,previous=None):
    previous=previous or {}
    generation=int(previous.get("learning_generation",0) or 0)+1
    univ=get_universe()
    cutoff=datetime.now(timezone.utc).date().isoformat()
    result={"created_utc":datetime.now(timezone.utc).isoformat(),
            "type":"direction3_10session_3pct",
            "target_precision_pct_each":80,
            "target_buy_success_pct":TARGET_WIN_PCT,
            "target_quality_requirements":{"mean_net_pct":"> 0","profit_factor":">= 1.1",
                "diagnostic_trades":">= 300","coverage_pct":">= 10"},
            "fixed_horizon_sessions":HORIZON,"sideways_band_pct":BAND_PCT,
            "evaluation_as_of":cutoff,
            "learning_generation":generation,
            "previous_report_created_utc":previous.get("created_utc"),
            "search_scope":"전체 완료 과거기간 반기별 순차평가 + 허용 라이선스 표준지표 + 방향·확률·손절·익절·보유기간·거래선별 변이",
            "research_only":True,"live_approved":False,
            "independent_holdout":False,
            "evaluation_warning":"전체 과거 구간을 수식 선택에 사용한 연구 적합도이며 미래 성공률 보장이 아닙니다.",
            "markets":{}}
    selected=os.environ.get("STOCK_AUDIT_MARKET","").strip().upper()
    history_start=os.environ.get("STOCK_AUDIT_HISTORY_START","").strip()
    # Long-history experiments sample every horizon to limit overlapping labels
    # and avoid multiplying the expensive plan-search workload by ten.
    history_step=10 if history_start else 2
    result["historical_request_start"]=history_start or "LIVE_DEFAULT_850_DAYS"
    result["signal_sampling_step_sessions"]=history_step
    markets=(selected,) if selected in ("KR","KQ","US") else ("KR","KQ","US")
    previous_markets=previous.get("markets") or {}
    for market in markets:
        previous_market=previous_markets.get(market) or {}
        previous_direction=((previous_market.get("tuning") or {})
                            .get("chosen") or None)
        previous_buy_model=previous_market.get("buy_success") or {}
        previous_buy=((previous_buy_model.get("tuning") or {})
                      .get("chosen") or None)
        previous_adaptive=previous_market.get("adaptive_plans") or {}
        previous_plan=((previous_adaptive.get("tuning") or {})
                       .get("chosen_plan") or None)
        plan_candidates=make_plan_candidates(generation,previous_plan)

        excluded={"KR":{"472150","0262B0"},"KQ":set(),"US":{"SOXL"}}[market]
        symbols=[x[0] for x in univ[market] if x[0] not in excluded][:per_market]
        if len(symbols)<per_market:
            failure={
                "learning_attempt_status":"RETAINED_PREVIOUS_DATA_FAILURE",
                "latest_attempt_loaded":0,
                "latest_attempt_error":(
                    f"{market} 구성종목 {len(symbols)}/{per_market}; "
                    "직전 검증 결과를 유지하고 다음 실행에서 재시도"),
            }
            if previous_market:
                result["markets"][market]={**previous_market,**failure}
            else:
                result["markets"][market]={
                    "loaded":0,"requested":per_market,"errors":[failure],
                    **failure}
            print("DIRECTION_MARKET_RETAIN "+market+" "+json.dumps(
                failure,ensure_ascii=False),flush=True)
            continue
        rows=[];buy_rows=[];plan_rows=[];loaded=0;errors=[]
        symbol_coverage={}
        for i,ticker in enumerate(symbols,1):
            try:
                data,source=(get_historical_prices(ticker,market,start_date=history_start)
                             if history_start else get_prices(ticker,market))
                x=compute(data)
                direction_rows=make_direction_rows(x,ticker,market,end_date=cutoff,
                                                   step=history_step)
                symbol_coverage[ticker]={"source":source,
                    "first":str(data.index[0].date()),
                    "last":str(data.index[-1].date()),
                    "price_sessions":len(data),
                    "labeled_samples":len(direction_rows)}
                rows.extend(direction_rows)
                buy_rows.extend(attach_buy_labels(x,direction_rows))
                plan_rows.extend(label_plans(
                    x,direction_rows,plans=plan_candidates))
                loaded+=1
            except Exception as exc:
                errors.append({"ticker":ticker,"error":str(exc)[:160]})
            if i%20==0:
                print(f"DIRECTION_PROGRESS {market} {i}/{len(symbols)} loaded={loaded} labels={len(rows)} plans={len(plan_candidates)}",flush=True)

        if (loaded<per_market or not rows or not buy_rows or not plan_rows):
            failure={
                "learning_attempt_status":"RETAINED_PREVIOUS_DATA_FAILURE",
                "latest_attempt_loaded":loaded,
                "latest_attempt_error":(
                    f"{market} 가격/라벨 부족: loaded={loaded}/{per_market}, "
                    f"direction={len(rows)}, buy={len(buy_rows)}, plans={len(plan_rows)}; "
                    "직전 검증 결과를 유지하고 다음 실행에서 재시도"),
                "latest_attempt_errors":errors[:10],
            }
            if previous_market:
                result["markets"][market]={**previous_market,**failure}
            else:
                result["markets"][market]={
                    "loaded":loaded,"requested":per_market,"errors":errors[:10],
                    **failure}
            print("DIRECTION_MARKET_RETAIN "+market+" "+json.dumps(
                failure,ensure_ascii=False),flush=True)
            continue

        metrics=score_directions(rows,eval_end=cutoff,
            search_generation=generation,previous_best=previous_direction)
        regime_challenger=score_regime_direction(
            rows,eval_end=cutoff,config=(metrics.get('tuning') or {}).get('chosen'))
        buy_metrics=score_buy_success(buy_rows,eval_end=cutoff,
            search_generation=generation,previous_best=previous_buy)
        adaptive_plans=score_plans(
            plan_rows,eval_end=cutoff,plans=plan_candidates,
            search_generation=generation,previous_best=previous_plan,
            target_success_pct=TARGET_WIN_PCT)
        result["markets"][market]={"loaded":loaded,"requested":len(symbols),
                                  "errors":errors[:10],
                                  "history_coverage_by_ticker":symbol_coverage,
                                  "labels_total":len(rows),
                                  "buy_labels_total":len(buy_rows),
                                  "buy_success":buy_metrics,
                                  "regime_challenger":regime_challenger,
                                  "adaptive_plans":adaptive_plans,**metrics}
        slim=metrics["windows"]["diagnostic"]
        print("DIRECTION_MARKET "+market+" "+json.dumps({
            "generation":generation,"loaded":loaded,"labels_total":len(rows),
            "buy_labels_total":len(buy_rows),
            "buy_diagnostic":buy_metrics["windows"]["diagnostic"],
            "adaptive_plan_diagnostic":adaptive_plans["windows"]["diagnostic"],
            "selective_challenger":adaptive_plans.get("selective_challenger"),
            "adaptive_plan_tuning":adaptive_plans["tuning"],
            "diagnostic":slim,
            "whole_history_direction":metrics["tuning"],
            "whole_history_buy":buy_metrics["tuning"],
            "trained_outcomes":metrics["trained_outcomes"]},
            ensure_ascii=False),flush=True)

    # A previous champion is a display-only fallback, NOT a fresh success.
    target_status, all_reached = certify_all(result, TARGET_WIN_PCT)
    result["target_quality_status"]=target_status
    result["target_quality_all_markets"]=all_reached
    result["training_health"]={m:market_health(v) for m,v in result["markets"].items()}
    result["all_markets_fresh"]=all(
        result["training_health"].get(m,{}).get("fresh_training",False)
        for m in ("KR","KQ","US"))
    result["deployment_gate"]="RESEARCH_ONLY_NOT_LIVE_APPROVED"
    bde_block=bde_evolution_block()
    if bde_block:result["bde_filter"]=bde_block
    result["evaluation_warning"]=(
        "기존 모델/과거 결과와 새 학습을 분리합니다. "
        "전체 과거를 수식 선택에 사용했으며 독립 실전 승률이 아닙니다.")
    # Compatibility aliases for older dashboard versions.
    result["target_50_status"]=target_status
    result["target_50_all_markets"]=result["target_quality_all_markets"]
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
