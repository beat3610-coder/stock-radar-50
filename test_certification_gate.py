"""Offline regression test: no impressive but previously inspected backtest may approve live sales."""
from certification import assess, wilson_lower

def sample():
    market={
        "requested":100,"loaded":100,
        "validation_accepted":True,"selected_for_diagnostic":"trend_pullback",
        "holdout_selected_metrics":{"trades":100,"win_rate_pct":80.0,"mean_net_pct":2.0,"profit_factor":3.0},
        "portfolio":{"valid":True,"total_return_pct":10,"max_drawdown_pct":5},
        "stress_portfolio":{"total_return_pct":8}
    }
    return {"markets":{key:dict(market) for key in ("KR","KQ","US")},
            "benchmark_verified":True,"survivorship_bias":False,
            "historical_data_previously_inspected":True}

def run():
    result=assess(sample())
    assert result["status"]=="NOT_APPROVED",result
    assert any("previously inspected" in reason for reason in result["reasons"])
    assert any("forward" in reason for reason in result["reasons"])
    assert any("market-data" in reason for reason in result["reasons"])
    assert wilson_lower(0,100)>=0 and wilson_lower(0,100)<0.45
    # Actual report schema must be interpreted, not silently substituted by an
    # old "holdout" field, and poor out-of-sample returns must always fail.
    report=sample()
    report["historical_data_previously_inspected"]=False
    report["markets"]["KR"]["holdout_selected_metrics"]["win_rate_pct"]=20.0
    failed=assess(report)
    assert any("KR: win rate below 60%" in r for r in failed["reasons"])
    print("PASS: certification refuses in-sample performance, stale evidence and failed holdouts")

if __name__=="__main__":
    run()
