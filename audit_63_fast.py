"""Leakage-controlled 63-instrument legacy-vs-model audit.

The holdout dates are fixed. All candidate families share one feature schema,
are scored prequentially, and model configuration/policy selection uses only
development observations. This file is research-only and never approves live use.
"""
import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from universe import get_universe, MANDATORY_KR, MANDATORY_US
from naver_api_universe import naver_api_top100
from providers import get_historical_prices
from strategy_lab import candidates_from_frame, TRAIN_END, VALID_END, HOLDOUT_END, PARAMS
from broad_candidates import broad_candidates_from_computed
from predictive_model import CONFIGS, DEFAULT_CONFIG, prepare_snapshot, predict_from_snapshot
from audit_300 import metrics

SAMPLE_START="2024-07-01"
MIN_HISTORY_BARS=250
MIN_END_DATE=(date.fromisoformat(HOLDOUT_END)+timedelta(days=12)).isoformat()
MAX_PORTFOLIO_POSITIONS=5
PORTFOLIO_COST_PCT=0.35
MIN_POLICY_TRADES=10
MIN_DEVELOPMENT_EDGE=0.0
EV_QUANTILES=(0.70,0.80,0.90)

def _period(trades,period):
    if period=="development":
        return [t for t in trades if SAMPLE_START<=t["signal_date"] and
                t["entry_date"]<TRAIN_END and t["exit_date"]<TRAIN_END]
    if period=="validation":
        return [t for t in trades if TRAIN_END<=t["entry_date"]<VALID_END and t["exit_date"]<VALID_END]
    return [t for t in trades if VALID_END<=t["entry_date"]<HOLDOUT_END and t["exit_date"]<HOLDOUT_END]

def _trade_metrics(trades):
    result=metrics(trades)
    result["max_drawdown_sequence_pct"]=result.get("max_drawdown_pct")
    return result

def _brier(rows,key):
    usable=[]
    for row in rows:
        try:
            p=float(row[key]);y=float(float(row["net_pct"])>0)
            if np.isfinite(p) and 0<=p<=1:usable.append((p,y))
        except (KeyError,TypeError,ValueError):pass
    if not usable:return {"n":0,"brier":None}
    return {"n":len(usable),"brier":round(float(np.mean([(p-y)**2 for p,y in usable])),6)}

def _calibration(rows,key="model_probability"):
    """Fixed 10-point reliability bins and expected calibration error."""
    samples=[]
    for row in rows:
        try:
            p=float(row[key]);y=float(float(row["net_pct"])>0)
            if np.isfinite(p) and 0<=p<=1:samples.append((p,y))
        except (KeyError,TypeError,ValueError):pass
    bins=[];ece=0.0;n=len(samples)
    for i in range(10):
        lo=i/10;hi=(i+1)/10
        group=[(p,y) for p,y in samples if lo<=p<(hi if i<9 else hi+1e-12)]
        avgp=float(np.mean([p for p,_ in group])) if group else None
        win=float(np.mean([y for _,y in group])) if group else None
        if group:ece+=len(group)/max(1,n)*abs(avgp-win)
        bins.append({"bin":f"{lo:.1f}-{hi:.1f}","n":len(group),
                     "mean_predicted_pct":round(100*avgp,2) if avgp is not None else None,
                     "observed_win_pct":round(100*win,2) if win is not None else None})
    return {"n":n,"ece":round(ece,6) if n else None,"reliability_bins":bins}

def _auc(rows,key="model_probability"):
    pairs=[]
    for row in rows:
        try:
            p=float(row[key]);y=int(float(row["net_pct"])>0)
            if np.isfinite(p):pairs.append((p,y))
        except (KeyError,TypeError,ValueError):pass
    labels=np.asarray([y for _,y in pairs],dtype=int)
    positives=int(labels.sum());negatives=len(labels)-positives
    if not positives or not negatives:return {"n":len(pairs),"positive_n":positives,"negative_n":negatives,"auroc":None}
    scores=np.asarray([p for p,_ in pairs],dtype=float)
    order=np.argsort(scores,kind="mergesort");ranks=np.empty(len(scores),dtype=float)
    i=0
    while i<len(order):
        j=i+1
        while j<len(order) and scores[order[j]]==scores[order[i]]:j+=1
        ranks[order[i:j]]=(i+1+j)/2.0
        i=j
    rank_sum=float(ranks[labels==1].sum())
    auc=(rank_sum-positives*(positives+1)/2)/(positives*negatives)
    return {"n":len(pairs),"positive_n":positives,"negative_n":negatives,"auroc":round(float(auc),6)}

def _probability_spread(rows,key="model_probability"):
    vals=[]
    for row in rows:
        try:
            value=float(row[key])
            if np.isfinite(value):vals.append(value)
        except (KeyError,TypeError,ValueError):pass
    if not vals:return {"n":0,"p10_pct":None,"p50_pct":None,"p90_pct":None}
    q=np.percentile(vals,[10,50,90])
    return {"n":len(vals),"p10_pct":round(float(q[0])*100,3),
            "p50_pct":round(float(q[1])*100,3),"p90_pct":round(float(q[2])*100,3)}

def _ev_spread(rows):
    vals=[]
    for row in rows:
        try:
            value=float(row["model_expected_net_pct"])
            if np.isfinite(value):vals.append(value)
        except (KeyError,TypeError,ValueError):pass
    if not vals:return {"n":0,"mean_pct":None,"p10_pct":None,"p50_pct":None,"p90_pct":None}
    q=np.percentile(vals,[10,50,90])
    return {"n":len(vals),"mean_pct":round(float(np.mean(vals)),4),
            "p10_pct":round(float(q[0]),4),"p50_pct":round(float(q[1]),4),"p90_pct":round(float(q[2]),4)}

def _family_score_diagnostics(scored,period):
    rows=_period(scored,period);out={}
    for family in ("legacy","broad"):
        group=[t for t in rows if t.get("candidate_family")==family]
        out[family]={"candidate_n":len(group),"realized":_trade_metrics(group),
                     "probability":_probability_spread(group),"expected_net_pct":_ev_spread(group)}
    return out

def _development_legacy_coefficient(candidates,config_name):
    if not config_name:return None
    train=[t for t in candidates if t.get("signal_date","")>=SAMPLE_START
           and t.get("entry_date","")<TRAIN_END and t.get("exit_date","")<TRAIN_END]
    snapshot=prepare_snapshot(train,config_name)
    if snapshot is None or "candidate_legacy" not in snapshot["features"]:return None
    beta=float(snapshot["beta"][1+snapshot["features"].index("candidate_legacy")])
    return {"training_candidates":snapshot["n"],"standardized_log_odds_coefficient":round(beta,6),
            "odds_ratio_legacy_vs_broad":round(float(np.exp(np.clip(beta,-20,20))),6),
            "interpretation":"positive means higher predicted win odds for legacy candidates, conditional on the fixed feature set"}

def _unselected_candidate_diagnostics(legacy_candidates,broad_candidates):
    result={}
    for family,rows in (("legacy",legacy_candidates),("broad",broad_candidates),
                        ("combined",legacy_candidates+broad_candidates)):
        result[family]={period:_trade_metrics(_period(rows,period))
                        for period in ("development","validation","diagnostic")}
    return result

def _prequential_scores(candidates,config_name):
    config=CONFIGS[config_name]
    ordered=sorted(candidates,key=lambda t:(t["signal_date"],t["ticker"],t["entry_date"],t.get("strategy","")))
    closed=sorted(candidates,key=lambda t:(t["exit_date"],t["ticker"],t["entry_date"],t.get("strategy","")))
    scored=[];history=[];j=0;active=None;snapshot=None;status={"candidate_n":len(candidates),"scored_n":0,"insufficient_history_n":0}
    for trade in ordered:
        signal=trade["signal_date"]
        if signal!=active:
            while j<len(closed) and closed[j]["exit_date"]<signal:
                history.append(closed[j]);j+=1
            snapshot=prepare_snapshot(history,config);active=signal
        prediction=predict_from_snapshot(snapshot,trade)
        if prediction is None:
            status["insufficient_history_n"]+=1
            continue
        scored.append({**trade,**{"model_"+k:v for k,v in prediction.items()},
                       "model_probability":prediction["predicted_win_pct"]/100.0,
                       "model_config":config_name})
        status["scored_n"]+=1
    return scored,status

def _policy_thresholds(scored):
    values=[float(t["model_expected_net_pct"]) for t in _period(scored,"development")
            if np.isfinite(float(t.get("model_expected_net_pct",np.nan)))]
    if not values:return {}
    return {f"ev_p{int(q*100)}":round(float(np.quantile(values,q)),6) for q in EV_QUANTILES}

def _accept(trade,policy_name,thresholds):
    if policy_name=="all_scored":return True
    cutoff=thresholds.get(policy_name)
    return cutoff is not None and float(trade["model_expected_net_pct"])>=cutoff

def _resolve_nonoverlapping(trades):
    """At most one open trade per ticker; choose highest decision-time EV."""
    ordered=sorted(trades,key=lambda t:(t["entry_date"],t["ticker"],
                 -float(t.get("model_expected_net_pct",t.get("entry_net_rr",0))),t.get("strategy","")))
    chosen=[];blocked_until={};seen_entry=set()
    for trade in ordered:
        ticker=trade["ticker"];entry=trade["entry_date"]
        key=(ticker,entry)
        if key in seen_entry or entry<=blocked_until.get(ticker,""):continue
        chosen.append(trade);seen_entry.add(key);blocked_until[ticker]=trade["exit_date"]
    return chosen

def _select_config(scored_by_config):
    result={}
    for name,rows in scored_by_config.items():
        result[name]={}
        for period in ("development","validation","diagnostic"):
            subset=_period(rows,period)
            result[name][period]={"auroc":_auc(subset),"brier":_brier(subset,"model_probability"),
                                  "probability_spread":_probability_spread(subset)}
    eligible=[(v["development"]["auroc"]["auroc"],name) for name,v in result.items()
              if v["development"]["auroc"]["auroc"] is not None
              and v["development"]["auroc"]["n"]>=MIN_POLICY_TRADES]
    if not eligible:return None,result,"INSUFFICIENT_DEVELOPMENT_RANKING_SAMPLE"
    _,chosen=max(eligible,key=lambda x:(x[0],x[1]))
    return chosen,result,"DEVELOPMENT_AUROC_SELECTED"

def _policy_results(scored,thresholds,period):
    pool=_period(scored,period);out={}
    for name,cutoff in thresholds.items():
        passed=[t for t in pool if _accept(t,name,thresholds)]
        selected=_resolve_nonoverlapping(passed)
        out[name]={"threshold_expected_net_pct":cutoff,"scored_candidates":len(pool),
                   "gate_passed_candidates":len(passed),
                   "acceptance_rate_pct":round(100*len(passed)/len(pool),2) if pool else None,
                   "selected_trades":_trade_metrics(selected)}
    return out

def _select_policy(scored,thresholds,raw_candidates=None):
    dev=_period(scored,"development");policy_table=_policy_results(scored,thresholds,"development")
    results={name:row["selected_trades"] for name,row in policy_table.items()}
    eligible=[(m["mean_net_pct"],name) for name,m in results.items()
              if m["trades"]>=MIN_POLICY_TRADES and m["mean_net_pct"] is not None
              and m["mean_net_pct"]>MIN_DEVELOPMENT_EDGE]
    if eligible:
        # Select by development net return among the predeclared EV quantiles only.
        _,selected=max(eligible,key=lambda x:(x[0],results[x[1]]["trades"],x[1]))
        return selected,policy_table,"DEVELOPMENT_EDGE_SELECTED"
    raw_dev=_trade_metrics(_period(raw_candidates if raw_candidates is not None else scored,"development"))
    if raw_dev["trades"]>=MIN_POLICY_TRADES and raw_dev["mean_net_pct"] is not None \
            and raw_dev["mean_net_pct"]<=MIN_DEVELOPMENT_EDGE:
        return None,policy_table,"NO_DEVELOPMENT_EDGE"
    return None,policy_table,"INSUFFICIENT_DEVELOPMENT_SAMPLE"

def _coverage(rows,requested):
    n=len({t["ticker"] for t in rows})
    return {"traded_symbols":n,"requested_symbols":requested,
            "coverage_pct":round(100*n/requested,2) if requested else None}

def _equal_weight_replay(trades,frames,start,end,cap=MAX_PORTFOLIO_POSITIONS,cost_pct=PORTFOLIO_COST_PCT):
    """Daily mark-to-market, no leverage, at most five equal initial slots."""
    start_ts=pd.Timestamp(start);end_ts=pd.Timestamp(end)
    close={}
    for ticker,frame in frames.items():
        if frame is None or frame.empty:continue
        close[ticker]=pd.Series(frame["Close"].astype(float).values,index=pd.to_datetime(frame.index).normalize())
    dates=sorted(set().union(*(set(s.index) for s in close.values()))) if close else []
    dates=[d for d in dates if start_ts<=d<end_ts]
    if not dates:return {"valid":False,"trades_executed":0,"max_drawdown_pct":None,"reason":"no portfolio dates"}
    entries={}
    for t in trades:
        if start<=t.get("entry_date","")<end and t.get("ticker") in close:
            entries.setdefault(pd.Timestamp(t["entry_date"]),[]).append(t)
    initial=100.0;slot=initial/cap;cash=initial;holdings={};last={};curve=[];executed=0;max_open=0
    fee=cost_pct/200.0
    for day in dates:
        candidates=sorted(entries.get(day,[]),key=lambda t:(-float(t.get("model_expected_net_pct",t.get("score",0))),t["ticker"]))
        for t in candidates:
            ticker=t["ticker"]
            if ticker in holdings or len(holdings)>=cap or cash<slot:continue
            entry=float(t["entry"])
            if not np.isfinite(entry) or entry<=0:continue
            # Each slot is an equal share of initial equity, inclusive of its
            # entry fee, so exactly five same-day orders can fit without leverage.
            shares=slot*(1-fee)/entry
            cash-=slot
            holdings[ticker]={"shares":shares,"trade":t,"entry":entry}
            executed+=1;max_open=max(max_open,len(holdings))
        for ticker,series in close.items():
            if day in series.index:
                px=float(series.loc[day])
                if np.isfinite(px) and px>0:last[ticker]=px
        for ticker,item in list(holdings.items()):
            trade=item["trade"]
            if pd.Timestamp(trade["exit_date"])<=day:
                exit_px=float(trade["exit_price"])
                cash+=item["shares"]*exit_px*(1-fee)
                del holdings[ticker]
        equity=cash+sum(item["shares"]*last.get(ticker,item["entry"]) for ticker,item in holdings.items())
        curve.append((str(day.date()),equity))
    vals=np.asarray([v for _,v in curve]);peak=np.maximum.accumulate(np.r_[initial,vals])[1:]
    dd=100*(1-vals/peak)
    return {"valid":True,"method":"equal initial weight, max 5 concurrent, daily close mark-to-market",
            "max_positions":cap,"trades_executed":executed,"max_concurrent_positions":max_open,
            "total_return_pct":round(100*(vals[-1]/initial-1),3),
            "max_drawdown_pct":round(float(max(0,dd.max())),3),"days":len(curve)}

def _metrics_with_portfolio(trades,requested,frames,period):
    start,end=(TRAIN_END,VALID_END) if period=="validation" else (VALID_END,HOLDOUT_END)
    trade=_trade_metrics(trades);port=_equal_weight_replay(trades,frames,start,end)
    trade.update(_coverage(trades,requested))
    trade["max_drawdown_portfolio_5_equal_pct"]=port.get("max_drawdown_pct")
    trade["portfolio_trades_executed"]=port.get("trades_executed",0)
    trade["portfolio_method"]=port.get("method",port.get("reason"))
    return trade

def _metric_delta(model,legacy):
    fields={"trades":"trades_delta","win_rate_pct":"win_rate_pp_delta",
            "mean_net_pct":"mean_net_pp_delta","profit_factor":"profit_factor_delta",
            "max_drawdown_sequence_pct":"sequence_mdd_pp_delta",
            "max_drawdown_portfolio_5_equal_pct":"portfolio_mdd_pp_delta",
            "coverage_pct":"coverage_pp_delta"}
    result={}
    for source,target in fields.items():
        a=model.get(source);b=legacy.get(source)
        result[target]=(round(a-b,4) if isinstance(a,(int,float)) and isinstance(b,(int,float)) else None)
    return result

def _overlap_report(legacy,model):
    key=lambda t:(t["ticker"],t["entry_date"])
    old={key(t):t for t in legacy};new={key(t):t for t in model}
    common=sorted(set(old)&set(new));old_only=[t for k,t in old.items() if k not in new]
    new_only=[t for k,t in new.items() if k not in old]
    paired=[{"ticker":k[0],"entry_date":k[1],"legacy_net_pct":old[k]["net_pct"],
             "model_net_pct":new[k]["net_pct"],
             "model_minus_legacy_net_pp":round(float(new[k]["net_pct"])-float(old[k]["net_pct"]),4),
             "model_minus_legacy_win_pp":100*int(float(new[k]["net_pct"])>0)-100*int(float(old[k]["net_pct"])>0)} for k in common]
    return {"legacy_only":{"n":len(old_only),"metrics":_trade_metrics(old_only)},
            "model_only":{"n":len(new_only),"metrics":_trade_metrics(new_only)},
            "both":{"n":len(common),"legacy_metrics":_trade_metrics([old[k] for k in common]),
                    "model_metrics":_trade_metrics([new[k] for k in common]),
                    "paired_mean_net_delta_pp":round(float(np.mean([x["model_minus_legacy_net_pp"] for x in paired])),4) if paired else None,
                    "paired_win_delta_pp":round(float(np.mean([x["model_minus_legacy_win_pp"] for x in paired])),2) if paired else None,
                    "pairs":paired}}

def _audit_universe():
    try:
        kr_rows,kr_source=naver_api_top100("KR");kq_rows,kq_source=naver_api_top100("KQ")
        kr=[(c,n,"KR",s) for c,n,s in kr_rows];kq=[(c,n,"KQ",s) for c,n,s in kq_rows]
        kr.extend((c,n,"KR",s) for c,n,s in MANDATORY_KR)
        us=[(c,n,"US",s) for c,n,s in MANDATORY_US]
        return {"KR":kr,"KQ":kq,"US":us,"KR_source":kr_source,"KQ_source":kq_source,"US_source":"SOXL 필수 단일 ETF"}
    except Exception as exc:
        print(f"AUDIT_UNIVERSE_NAVER_FALLBACK {type(exc).__name__}: {str(exc)[:180]}",flush=True)
        return get_universe(include_us=False)

def _tag_candidates(legacy_by_variant,broad):
    all_legacy=[]
    for strategy,items in legacy_by_variant.items():
        for trade in items:
            all_legacy.append({**trade,"candidate_family":"legacy","candidate_legacy":1.0,
                               "candidate_variant":strategy})
    tagged_broad=[{**trade,"candidate_family":"broad","candidate_legacy":0.0,
                   "candidate_variant":"broad_market_ml"} for trade in broad]
    return all_legacy+tagged_broad,all_legacy,tagged_broad

def run():
    universe=_audit_universe()
    cohorts={"KOSPI":universe["KR"][:30],"KOSDAQ":universe["KQ"][:30],
             "TIGER_ETF":universe["KR"][-2:],"SOXL":universe["US"][-1:]}
    out={"created_utc":datetime.now(timezone.utc).isoformat(),"status":"RESEARCH_ONLY_NOT_LIVE_APPROVED",
         "sample_start":SAMPLE_START,"windows":{"development_end":TRAIN_END,"validation_end":VALID_END,"diagnostic_end":HOLDOUT_END},
         "cost_round_trip_pct":0.35,"portfolio":{"max_concurrent_positions":MAX_PORTFOLIO_POSITIONS,
         "weighting":"equal initial notional slots","mdd":"daily mark-to-market"},
         "universe_sources":{k:universe.get(k+"_source") for k in ("KR","KQ","US")},"cohorts":{}}
    all_data_complete=True
    for cohort,items in cohorts.items():
        rows=[];legacy_by_variant={k:[] for k in PARAMS};broad_pool=[];frames={};errors=[]
        for i,(ticker,name,market,sector) in enumerate(items,1):
            try:
                minimum_bars=1 if ticker=="0262B0" else MIN_HISTORY_BARS
                prices,source=get_historical_prices(ticker,market,min_bars=minimum_bars)
                first=str(prices.index[0].date());last=str(prices.index[-1].date());post_period_listing=first>HOLDOUT_END
                if len(prices)<MIN_HISTORY_BARS and not post_period_listing:raise ValueError(f"history {len(prices)} bars < {MIN_HISTORY_BARS}")
                if last<MIN_END_DATE:raise ValueError(f"OHLCV ends {last}, before required {MIN_END_DATE}")
                computed,variants=candidates_from_frame(prices,ticker,market)
                for strategy,trades in variants.items():legacy_by_variant[strategy].extend(t for t in trades if t["signal_date"]>=SAMPLE_START)
                broad=[t for t in broad_candidates_from_computed(computed,ticker,market,end_date=HOLDOUT_END,cost_pct=0.35) if t["signal_date"]>=SAMPLE_START]
                broad_pool.extend(broad);frames[ticker]=prices
                rows.append({"ticker":ticker,"name":name,"source":source,"bars":len(prices),"first_date":first,"last_date":last,
                    "history_status":"LISTED_AFTER_DIAGNOSTIC_WINDOW_NO_OOS_HISTORY" if post_period_listing else "HISTORICAL_WINDOW_AVAILABLE",
                    "legacy_variants":variants,"broad_candidates":broad})
            except Exception as exc:errors.append({"ticker":ticker,"name":name,"error":str(exc)[:240]})
            print(f"AUDIT_PROGRESS {cohort} {i}/{len(items)} loaded={len(rows)}",flush=True)

        legacy_dev={name:_trade_metrics(_period(trades,"development")) for name,trades in legacy_by_variant.items()}
        legacy_eligible=[name for name,m in legacy_dev.items() if m["trades"]>=MIN_POLICY_TRADES and m["mean_net_pct"] is not None]
        legacy_selected=max(legacy_eligible,key=lambda name:(legacy_dev[name]["mean_net_pct"],name),default=None)
        legacy_trades=_resolve_nonoverlapping(legacy_by_variant.get(legacy_selected,[])) if legacy_selected else []
        combined,legacy_candidates,broad_candidates=_tag_candidates(legacy_by_variant,broad_pool)
        scored_by_config={};score_status={}
        for config_name in CONFIGS:
            scored_by_config[config_name],score_status[config_name]=_prequential_scores(combined,config_name)
        selected_config,config_development,config_status=_select_config(scored_by_config)
        scored=scored_by_config[selected_config] if selected_config else []
        thresholds=_policy_thresholds(scored)
        selected_policy,policy_development,policy_status=_select_policy(scored,thresholds,combined)
        dev_population=_trade_metrics(_resolve_nonoverlapping(_period(combined,"development")))
        raw_dev=_trade_metrics(_period(combined,"development"))
        no_edge=policy_status=="NO_DEVELOPMENT_EDGE"
        model_trades=_resolve_nonoverlapping([t for t in scored if _accept(t,selected_policy,thresholds)]) if selected_policy else []
        cohort_result={"requested":len(items),"loaded":len(rows),"errors":errors,
          "data_coverage_pct":round(100*len(rows)/len(items),2) if items else None,
          "data_complete":len(rows)==len(items) and not errors,
          "feature_schema":{"full":"all 17 fixed-scale features","core":"12 fixed-scale features",
                            "payoff_features":["entry_net_rr","entry_net_upside_pct","entry_net_downside_pct","signal_stop_atr"],
                            "family_feature":"candidate_legacy"},
          "candidate_pool":{"legacy_variant_candidates":len(legacy_candidates),"broad_candidates":len(broad_candidates),
                            "combined_candidates":len(combined)},
          "legacy_selected_development_strategy":legacy_selected,"legacy_development_by_variant":legacy_dev,
          "selected_model_config":selected_config,"model_config_status":config_status,
          "model_config_selection_metric":"development prequential AUROC; Brier is calibration diagnostic only",
          "model_config_scores_by_period":config_development,"model_config_score_status":score_status,
          "selected_config_candidate_legacy_coefficient_at_development_end":_development_legacy_coefficient(combined,selected_config),
          "selected_model_policy":selected_policy,"model_policy_status":policy_status,
          "model_policy_thresholds_expected_net_pct":thresholds,
          "model_policy_development":policy_development,"development_population":dev_population,
          "unselected_candidate_diagnostics":_unselected_candidate_diagnostics(legacy_candidates,broad_candidates),
            "no_development_edge_gate":{"status":"NO_DEVELOPMENT_EDGE" if no_edge else policy_status,
              "threshold_mean_net_pct":0.0,"development_candidate_n":raw_dev["trades"],
              "development_candidate_mean_net_pct":raw_dev["mean_net_pct"]},
          "periods":{},"candidate_family_score_diagnostics":{},"instruments":[]}
        cohort_result["candidate_family_score_diagnostics"]={period:_family_score_diagnostics(scored,period)
            for period in ("development","validation","diagnostic")}
        for period in ("validation","diagnostic"):
            old=_resolve_nonoverlapping(_period(legacy_trades,period));new=_resolve_nonoverlapping(_period(model_trades,period))
            oldm=_metrics_with_portfolio(old,len(items),frames,period);newm=_metrics_with_portfolio(new,len(items),frames,period)
            candidates=_period(scored,period)
            train=_period(combined,"development")
            base=(sum(float(t["net_pct"])>0 for t in train)+2)/(len(train)+4) if train else None
            base_rows=[{**t,"base_probability":base} for t in candidates] if base is not None else []
            brier_base=_brier(base_rows,"base_probability");brier_model=_brier(candidates,"model_probability")
            calibration=_calibration(candidates)
            policy_period=_policy_results(scored,thresholds,period)
            delta=_metric_delta(newm,oldm)
            cohort_result["periods"][period]={"legacy":oldm,"model":newm,"delta_model_minus_legacy":delta,
                "paired_comparison":_overlap_report(old,new),
                "calibration":{"candidate_sample_n":brier_model["n"],"base_brier":brier_base["brier"],
                    "model_brier":brier_model["brier"],"brier_improvement_base_minus_model":
                       round(brier_base["brier"]-brier_model["brier"],6) if brier_base["brier"] is not None and brier_model["brier"] is not None else None,
                    "auroc":_auc(candidates),"probability_spread":_probability_spread(candidates),**calibration},
                "policy_diagnostics":policy_period}
        for row in rows:
            old=row["legacy_variants"].get(legacy_selected,[]) if legacy_selected else []
            new=[t for t in model_trades if t["ticker"]==row["ticker"]]
            cohort_result["instruments"].append({"ticker":row["ticker"],"name":row["name"],"source":row["source"],
              "bars":row["bars"],"first_date":row["first_date"],"last_date":row["last_date"],"history_status":row["history_status"],
              "broad_candidates":{"development":_trade_metrics(_period(row["broad_candidates"],"development")),
                  "validation":_trade_metrics(_period(row["broad_candidates"],"validation")),"diagnostic":_trade_metrics(_period(row["broad_candidates"],"diagnostic"))},
              "legacy":{"validation":_trade_metrics(_period(old,"validation")),"diagnostic":_trade_metrics(_period(old,"diagnostic"))},
              "model":{"validation":_trade_metrics(_period(new,"validation")),"diagnostic":_trade_metrics(_period(new,"diagnostic"))}})
        out["cohorts"][cohort]=cohort_result
        all_data_complete=all_data_complete and cohort_result["data_complete"]
        print("AUDIT_RESULT "+json.dumps({"cohort":cohort,"requested":len(items),"loaded":len(rows),
          "legacy_strategy":legacy_selected,"model_config":selected_config,"model_policy":selected_policy,
          "model_policy_status":policy_status,"validation":cohort_result["periods"]["validation"],
          "diagnostic":cohort_result["periods"]["diagnostic"]},ensure_ascii=False),flush=True)
    out["data_integrity_status"]="PASS" if all_data_complete else "FAIL"
    Path("data").mkdir(exist_ok=True)
    Path("data/audit-63-latest.json").write_text(json.dumps(out,ensure_ascii=False,indent=2,default=str),encoding="utf-8")
    if not all_data_complete:raise SystemExit("FAILED: incomplete 63-instrument OHLCV coverage; see data/audit-63-latest.json")
    return out

if __name__=="__main__":run()
