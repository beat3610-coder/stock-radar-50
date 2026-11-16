"""Research-only quality gates. Fail closed on stale training or missing data."""
from datetime import datetime, timezone
import math

MARKETS=("KR","KQ","US")

def finite_number(value):
    if isinstance(value,bool) or value is None:
        return None
    try:
        v=float(value)
        return v if math.isfinite(v) else None
    except (TypeError,ValueError):
        return None

def market_health(data):
    """The previous champion may be displayed but may not certify a failed run."""
    requested=int(data.get("requested") or 0)
    loaded=int(data.get("latest_attempt_loaded",data.get("loaded",0)) or 0)
    errors=data.get("errors") or []
    failure=str(data.get("learning_attempt_status") or "")
    source_failure=failure.startswith("RETAINED_")
    complete=requested>=100 and loaded==requested and not source_failure
    return {
        "fresh_training":bool(complete),
        "requested":requested,
        "loaded":loaded,
        "coverage_pct":round(100*loaded/requested,2) if requested else 0,
        "retained_previous_model":source_failure,
        "errors_count":len(errors),
        "reason":("구성종목·시세 불완전/기존 성적 유지" if not complete
                  else "종목 수집 완료(독립 검증 아님)"),
    }

def certify_market(data, target_win_pct=60):
    """Only a research objective check; never a live trading authorization."""
    health=market_health(data)
    challenger=((data.get("adaptive_plans") or {}).get("selective_challenger") or {})
    d=((challenger.get("windows") or {}).get("diagnostic") or {})
    win=finite_number(d.get("actual_win_pct"))
    net=finite_number(d.get("mean_net_pct"))
    pf=finite_number(d.get("profit_factor"))
    trades=finite_number(d.get("trades"))
    coverage=finite_number(d.get("coverage_pct"))
    validation=bool(challenger.get("validation_eligible"))
    conditions={
        "fresh_training":health["fresh_training"],
        "validation_eligible":validation,
        "win_rate_reached":win is not None and win>=target_win_pct,
        "positive_net_reached":net is not None and net>0,
        "profit_factor_reached":pf is not None and pf>=1.1,
        "sample_reached":trades is not None and trades>=300,
        "coverage_reached":coverage is not None and coverage>=10,
    }
    return {"actual_win_pct":win,"mean_net_pct":net,"profit_factor":pf,
            "trades":trades,"coverage_pct":coverage,
            "validation_eligible":validation, **conditions,
            "reached":all(conditions.values()),
            "live_approved":False,
            "independent_holdout":False,
            "data_health":health}

def certify_all(report,target_win_pct=60):
    markets=report.get("markets") or {}
    states={m:certify_market(markets.get(m) or {},target_win_pct)
            for m in MARKETS}
    return states,all(s["reached"] for s in states.values())
