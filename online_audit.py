"""Standalone resource-bounded 300-stock predictive-calibration audit.

No retrospective strategy selection is performed. Score ALL executable broad
candidate trades; independently report probability calibration vs market-only
baseline and the three frozen research selection policies. Live approval OFF.
"""
from datetime import datetime,timezone
from engine import compute
from providers import get_prices
from universe import get_universe
from broad_candidates import broad_candidates_from_computed
from online_edge import score_stream,POLICIES
from audit_300 import metrics

TRAIN_END="2025-11-01"
VALID_END="2026-04-01"
END="2026-11-01"

def window(trades,part):
    def belongs(t):
        e=t["entry_date"];x=t["exit_date"]
        if part=="train":return e<TRAIN_END and x<TRAIN_END
        if part=="validation":return TRAIN_END<=e<VALID_END and x<VALID_END
        return VALID_END<=e<END and x<END
    return [t for t in trades if belongs(t)]

def run_online(per_market=100):
    univ=get_universe()
    report={"created_utc":datetime.now(timezone.utc).isoformat(),
            "type":"v9_all_candidate_probability_score",
            "goal_win_pct":70,"live_approved":False,
            "method":"historical market-specific prior-only online probability; all candidates scored",
            "caveats":["2026 evaluation is repeatedly examined, not a fresh independent holdout",
                       "Current surviving stock constituents only",
                       "Daily stop-first OHLC execution and estimated 0.35% costs",
                       "Probability quality vs naive market base is primary; selecting 70% after seeing tests is prohibited"],
            "markets":{}}
    for market in ("KR","KQ","US"):
        exclude={"KR":{"472150","0262B0"},"KQ":set(),"US":{"SOXL"}}[market]
        items=[t for t in univ[market] if t[0] not in exclude][:per_market]
        pool=[];errors=[];loaded=0
        for i,(ticker,*_) in enumerate(items,1):
            try:
                raw,_=get_prices(ticker,market)
                computed=compute(raw)
                pool.extend(broad_candidates_from_computed(computed,ticker,market,
                           end_date=END,cost_pct=0.35))
                loaded+=1
            except Exception as exc:
                errors.append({"ticker":ticker,"reason":str(exc)[:140]})
            if i%10==0:print(f"ONLINE_PROGRESS {market} {i}/{len(items)} loaded={loaded} candidate_count={len(pool)}",flush=True)
        scored,details=score_stream(pool,market)
        policy_results={}
        for policy,trades in scored.items():
            policy_results[policy]={part:metrics(window(trades,part))
                for part in ("train","validation","diagnostic")}
        base_results={part:metrics(window(pool,part))
                      for part in ("train","validation","diagnostic")}
        summary={"requested":len(items),"loaded":loaded,"error_count":len(errors),
                 "errors":errors[:5],"candidates":len(pool),
                 "scoring":details,"all_candidates":base_results,
                 "policies":policy_results}
        report["markets"][market]=summary
        import json
        print("ONLINE_MARKET "+market+" "+json.dumps({
            "loaded":loaded,"candidates":len(pool),
            "by_window":details["by_window"],
            "probability_quality":details["probability_quality"],
            "policies":{k:v["diagnostic"] for k,v in policy_results.items()}},
            ensure_ascii=False,default=str),flush=True)
    return report

if __name__=="__main__":
    import json
    print("ONLINE300_RESULT "+json.dumps(run_online(),ensure_ascii=False,default=str),flush=True)
