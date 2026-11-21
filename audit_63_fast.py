"""Out-of-sample audit for 30 KOSPI + 30 KOSDAQ + 2 TIGER ETFs + SOXL.

The legacy signal is kept as the baseline. A broader, feature-based candidate
model is fit prequentially using only trades closed before each signal date.
Policy selection uses the development interval only. Validation and diagnostic
figures are reported even when negative; this program never approves live use.
"""
import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from universe import get_universe
from universe import MANDATORY_KR, MANDATORY_US
from naver_api_universe import naver_api_top100
from providers import get_historical_prices
from strategy_lab import (candidates_from_frame, TRAIN_END, VALID_END,
                          HOLDOUT_END, PARAMS)
from broad_candidates import broad_candidates_from_computed
from predictive_model import (POLICIES, prepare_snapshot,
                              predict_from_snapshot)
from audit_300 import metrics

# The prior 63-symbol run used providers.get_prices' rolling 850-day range.
# A fixed floor retains a comparable development/test window while the longer
# history is now used only to initialize indicators and fill the OHLCV cache.
SAMPLE_START = "2024-07-01"
MIN_HISTORY_BARS = 250
MIN_END_DATE = (date.fromisoformat(HOLDOUT_END)+timedelta(days=12)).isoformat()

def _period(trades, period):
    if period == "development":
        return [t for t in trades if SAMPLE_START<=t["signal_date"] and
                t["entry_date"]<TRAIN_END and t["exit_date"]<TRAIN_END]
    if period == "validation":
        return [t for t in trades if TRAIN_END<=t["entry_date"]<VALID_END and
                t["exit_date"]<VALID_END]
    return [t for t in trades if VALID_END<=t["entry_date"]<HOLDOUT_END and
            t["exit_date"]<HOLDOUT_END]

def _brier(rows, probability_key):
    usable=[]
    for row in rows:
        try:
            p=float(row[probability_key]); y=float(float(row["net_pct"])>0)
            if np.isfinite(p):usable.append((p,y))
        except (KeyError,TypeError,ValueError):
            continue
    if not usable:return {"n":0,"brier":None}
    return {"n":len(usable),"brier":round(float(np.mean([(p-y)**2 for p,y in usable])),6)}

def _prequential_scores(candidates):
    ordered=sorted(candidates,key=lambda t:(t["signal_date"],t["ticker"],t["entry_date"]))
    closed=sorted(candidates,key=lambda t:(t["exit_date"],t["ticker"],t["entry_date"]))
    scored=[];history=[];j=0;active=None;snapshot=None
    for trade in ordered:
        signal=trade["signal_date"]
        if signal!=active:
            while j<len(closed) and closed[j]["exit_date"]<signal:
                history.append(closed[j]);j+=1
            snapshot=prepare_snapshot(history)
            active=signal
        probability=predict_from_snapshot(snapshot,trade)
        method="regularized_logistic"
        if probability is not None:
            pass
        else:
            # Sparse cohorts (for example one ETF) still receive an honest
            # prior-only score from already closed trades; they are explicitly
            # labelled as fallback and are never treated as a fitted model.
            wins=[float(t["net_pct"]) for t in history if float(t["net_pct"])>0]
            losses=[float(t["net_pct"]) for t in history if float(t["net_pct"])<=0]
            n=len(history);p=(len(wins)+2)/(n+4)
            mean_win=float(np.mean(wins)) if wins else 0.0
            mean_loss=float(np.mean(losses)) if losses else 0.0
            expected=p*mean_win+(1-p)*mean_loss
            probability={"training_completed":n,"predicted_win_pct":round(100*p,2),
                         "baseline_win_pct":round(100*p,2),
                         "expected_net_pct":round(expected,3),
                         "predicted_uplift_pct":0.0}
            method="closed_trade_prior_only"
        scored.append({**trade,**{"model_"+k:v for k,v in probability.items()},
                       "model_probability":probability["predicted_win_pct"]/100.0,
                       "model_method":method})
    return scored

def _policy_accept(trade, policy_name):
    if policy_name=="all_scored":return True
    policy=POLICIES[policy_name]
    return (trade["model_predicted_win_pct"]>=policy["win_pct"] and
            trade["model_predicted_uplift_pct"]>=policy["uplift_pct"] and
            trade["model_expected_net_pct"]>=policy["expected_net_pct"])

def _select_policy(scored):
    development=_period(scored,"development")
    results={}
    for name in ("all_scored",*POLICIES):
        chosen=[t for t in development if _policy_accept(t,name)]
        results[name]=metrics(chosen)
    eligible=[name for name,m in results.items() if m["trades"]>=10 and m["mean_net_pct"] is not None]
    # A negative development result is still reported; it is not replaced by
    # an unvalidated zero-trade claim. No validation/diagnostic selection.
    selected=max(eligible,key=lambda n:results[n]["mean_net_pct"],default=None)
    if selected is not None:
        status="DEVELOPMENT_SELECTED"
    elif results["all_scored"]["trades"]:
        selected="all_scored"
        status="INSUFFICIENT_DEVELOPMENT_SAMPLE_FALLBACK_ALL_SCORED"
    else:
        status="NO_SCORED_CANDIDATES"
    return selected,results,status

def _coverage(rows, requested):
    symbols={t["ticker"] for t in rows}
    return {"traded_symbols":len(symbols),"requested_symbols":requested,
            "coverage_pct":round(100*len(symbols)/requested,2) if requested else None}

def _audit_universe():
    """Use the market-qualified ranked feed directly; sectors are irrelevant here."""
    try:
        kr_rows,kr_source=naver_api_top100("KR")
        kq_rows,kq_source=naver_api_top100("KQ")
        kr=[(code,name,"KR",sector) for code,name,sector in kr_rows]
        kq=[(code,name,"KQ",sector) for code,name,sector in kq_rows]
        kr.extend((code,name,"KR",sector) for code,name,sector in MANDATORY_KR)
        us=[(code,name,"US",sector) for code,name,sector in MANDATORY_US]
        return {"KR":kr,"KQ":kq,"US":us,
                "KR_source":kr_source,"KQ_source":kq_source,
                "US_source":"SOXL 필수 단일 ETF"}
    except Exception as exc:
        print(f"AUDIT_UNIVERSE_NAVER_FALLBACK {type(exc).__name__}: {str(exc)[:180]}",flush=True)
        return get_universe(include_us=False)

def run():
    universe=_audit_universe()
    cohorts={
        "KOSPI":universe["KR"][:30],
        "KOSDAQ":universe["KQ"][:30],
        "TIGER_ETF":universe["KR"][-2:],
        "SOXL":universe["US"][-1:],
    }
    out={"created_utc":datetime.now(timezone.utc).isoformat(),
         "status":"RESEARCH_ONLY_NOT_LIVE_APPROVED",
         "sample_start":SAMPLE_START,
         "windows":{"development_end":TRAIN_END,"validation_end":VALID_END,
                    "diagnostic_end":HOLDOUT_END},
         "cost_round_trip_pct":0.35,
         "universe_sources":{k:universe.get(k+"_source") for k in ("KR","KQ","US")},
         "selection":"development-period highest mean net return among policies with >=10 trades; validation and diagnostic excluded",
         "cohorts":{}}
    all_data_complete=True
    for cohort,items in cohorts.items():
        rows=[];legacy_pool={k:[] for k in PARAMS};broad_pool=[];errors=[]
        for i,(ticker,name,market,sector) in enumerate(items,1):
            try:
                # A0262B0 began trading after the fixed diagnostic cutoff.
                # Keep its real short history for an explicit out-of-window
                # status; do not synthesize pre-listing returns.
                minimum_bars=1 if ticker=="0262B0" else MIN_HISTORY_BARS
                prices,source=get_historical_prices(ticker,market,min_bars=minimum_bars)
                first=str(prices.index[0].date());last=str(prices.index[-1].date())
                post_period_listing=first>HOLDOUT_END
                if len(prices)<MIN_HISTORY_BARS and not post_period_listing:
                    raise ValueError(f"history {len(prices)} bars < {MIN_HISTORY_BARS}")
                if last<MIN_END_DATE:
                    raise ValueError(f"OHLCV ends {last}, before required {MIN_END_DATE}")
                computed,variants=candidates_from_frame(prices,ticker,market)
                for strategy,trades in variants.items():
                    legacy_pool[strategy].extend(t for t in trades if t["signal_date"]>=SAMPLE_START)
                broad=broad_candidates_from_computed(computed,ticker,market,
                                                     end_date=HOLDOUT_END,cost_pct=0.35)
                broad=[t for t in broad if t["signal_date"]>=SAMPLE_START]
                broad_pool.extend(broad)
                rows.append({"ticker":ticker,"name":name,"market":market,
                             "source":source,"bars":len(prices),"first_date":first,
                             "last_date":last,
                             "history_status":("LISTED_AFTER_DIAGNOSTIC_WINDOW_NO_OOS_HISTORY"
                                if post_period_listing else "HISTORICAL_WINDOW_AVAILABLE"),
                             "legacy_variants":variants,
                             "broad_candidates":broad})
            except Exception as exc:
                errors.append({"ticker":ticker,"name":name,"error":str(exc)[:240]})
            print(f"AUDIT_PROGRESS {cohort} {i}/{len(items)} loaded={len(rows)}",flush=True)

        # Select the best legacy exit/entry variant on development only.
        legacy_dev={key:metrics(_period(trades,"development"))
                    for key,trades in legacy_pool.items()}
        eligible=[k for k,m in legacy_dev.items() if m["trades"]>=10 and m["mean_net_pct"] is not None]
        legacy_selected=max(eligible,key=lambda k:legacy_dev[k]["mean_net_pct"],default=None)
        legacy_trades=legacy_pool.get(legacy_selected,[]) if legacy_selected else []

        # The probability model sees the full broad candidate pool; its prior
        # and logistic features are updated only after earlier trades exit.
        train=_period(broad_pool,"development")
        prior=(sum(float(t["net_pct"])>0 for t in train)+2)/(len(train)+4) if train else None
        scored=_prequential_scores(broad_pool)
        selected_policy,policy_development,policy_status=_select_policy(scored)
        model_trades=([t for t in scored if _policy_accept(t,selected_policy)]
                      if selected_policy else [])
        cohort_result={"requested":len(items),"loaded":len(rows),"errors":errors,
            "data_coverage_pct":round(100*len(rows)/len(items),2) if items else None,
            "data_complete":len(rows)==len(items) and not errors,
            "broad_candidate_trades":{"development":metrics(_period(broad_pool,"development")),
                                      "validation":metrics(_period(broad_pool,"validation")),
                                      "diagnostic":metrics(_period(broad_pool,"diagnostic"))},
            "legacy_selected_development_strategy":legacy_selected,
            "legacy_development":legacy_dev.get(legacy_selected),
            "selected_model_policy":selected_policy,
            "model_policy_status":policy_status,
            "model_policy_development":policy_development,
            "model_development_prior_win_probability":round(prior,6) if prior is not None else None,
            "periods":{},"instruments":[]}
        for period in ("validation","diagnostic"):
            legacy_set=_period(legacy_trades,period)
            candidate_set=_period(scored,period)
            model_set=_period(model_trades,period)
            brier_base=(_brier(candidate_set,"base_probability") if candidate_set else {"n":0,"brier":None})
            # One fixed empirical training probability is the calibrated base
            # comparator; all candidates and periods are scored on the same rows.
            if prior is not None:
                candidate_set=[{**t,"base_probability":prior} for t in candidate_set]
                brier_base=_brier(candidate_set,"base_probability")
            else:brier_base={"n":0,"brier":None}
            brier_model=_brier(candidate_set,"model_probability")
            cohort_result["periods"][period]={
                "legacy":{**metrics(legacy_set),**_coverage(legacy_set,len(items))},
                "model":{**metrics(model_set),**_coverage(model_set,len(items))},
                "brier":{"candidate_sample_n":brier_model["n"],
                         "base":brier_base["brier"],"model":brier_model["brier"],
                         "improvement_base_minus_model":(round(brier_base["brier"]-brier_model["brier"],6)
                            if brier_base["brier"] is not None and brier_model["brier"] is not None else None)}}
        for row in rows:
            old=[]
            if legacy_selected:old=row["legacy_variants"].get(legacy_selected,[])
            selected_keys={(t["ticker"],t["entry_date"],t["strategy"]) for t in model_trades}
            new=[t for t in row["broad_candidates"]
                 if (t["ticker"],t["entry_date"],t["strategy"]) in selected_keys]
            cohort_result["instruments"].append({"ticker":row["ticker"],"name":row["name"],
                "source":row["source"],"bars":row["bars"],"first_date":row["first_date"],"last_date":row["last_date"],
                "history_status":row["history_status"],
                "broad_candidates":{"development":metrics(_period(row["broad_candidates"],"development")),
                                     "validation":metrics(_period(row["broad_candidates"],"validation")),
                                     "diagnostic":metrics(_period(row["broad_candidates"],"diagnostic"))},
                "legacy":{"validation":metrics(_period(old,"validation")),
                          "diagnostic":metrics(_period(old,"diagnostic"))},
                "model":{"validation":metrics(_period(new,"validation")),
                         "diagnostic":metrics(_period(new,"diagnostic"))}})
        out["cohorts"][cohort]=cohort_result
        all_data_complete=all_data_complete and cohort_result["data_complete"]
        print("AUDIT_RESULT "+json.dumps({"cohort":cohort,
            "requested":len(items),"loaded":len(rows),
            "legacy_strategy":legacy_selected,"model_policy":selected_policy,
            "validation":cohort_result["periods"]["validation"],
            "diagnostic":cohort_result["periods"]["diagnostic"]},ensure_ascii=False),flush=True)
    out["data_integrity_status"]="PASS" if all_data_complete else "FAIL"
    out["status"]="RESEARCH_ONLY_NOT_LIVE_APPROVED"
    Path("data").mkdir(exist_ok=True)
    Path("data/audit-63-latest.json").write_text(
        json.dumps(out,ensure_ascii=False,indent=2,default=str),encoding="utf-8")
    if not all_data_complete:
        raise SystemExit("FAILED: incomplete 63-instrument OHLCV coverage; see data/audit-63-latest.json")
    return out

if __name__=="__main__":
    run()
