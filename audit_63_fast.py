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
    train.sort(key=lambda t:(t.get("exit_date",""),t.get("ticker",""),t.get("entry_date","")))
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
        by_family={}
        for family in ("legacy","broad"):
            family_pool=[t for t in pool if t.get("candidate_family")==family]
            family_passed=[t for t in passed if t.get("candidate_family")==family]
            by_family[family]={"scored_candidates":len(family_pool),"gate_passed_candidates":len(family_passed),
                "acceptance_rate_pct":round(100*len(family_passed)/len(family_pool),2) if family_pool else None,
                "selected_trades":_trade_metrics(_resolve_nonoverlapping(family_passed))}
        out[name]={"threshold_expected_net_pct":cutoff,"scored_candidates":len(pool),
                   "gate_passed_candidates":len(passed),
                   "acceptance_rate_pct":round(100*len(passed)/len(pool),2) if pool else None,
                   "selected_trades":_trade_metrics(selected),"by_family":by_family}
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

def _run_pre_r_rank():
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
        config_policy_diagnostics={}
        for config_name,config_scored in scored_by_config.items():
            config_thresholds=_policy_thresholds(config_scored)
            config_policy_diagnostics[config_name]={"development_thresholds_only":config_thresholds,
                "periods":{period:_policy_results(config_scored,config_thresholds,period)
                           for period in ("development","validation","diagnostic")}}
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
          "model_config_scores_by_period":config_development,"model_config_policy_diagnostics":config_policy_diagnostics,
          "model_config_score_status":score_status,
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

SAMPLE_START="2016-01-01"

from r_rank_model import (MODEL_CONFIGS, MIN_TRAIN_SAMPLES_BY_COHORT, STRATEGIES, accepts as r_accepts,
    make_variant_thresholds, realized_r, walk_forward_scores)
from scipy.stats import spearmanr

def _rank_metrics(rows):
    valid=[r for r in rows if np.isfinite(float(r.get("realized_r",np.nan)))
           and np.isfinite(float(r.get("model_predicted_r",np.nan)))]
    if not valid:return {"n":0,"r_ic_spearman":None,"tercile_top_minus_bottom_realized_r":None,
                         "top20_realized_r":None,"top20_mean_net_pct":None}
    y=np.asarray([float(r["realized_r"]) for r in valid]);p=np.asarray([float(r["model_predicted_r"]) for r in valid])
    ic=spearmanr(p,y).statistic if len(valid)>2 else None
    order=np.argsort(p,kind="mergesort");k=max(1,len(valid)//3)
    top=float(y[order[-k:]].mean());bottom=float(y[order[:k]].mean())
    n20=max(1,int(np.ceil(len(valid)*.20)));ix=order[-n20:]
    r_ic=round(float(ic),6) if ic is not None and np.isfinite(ic) else None
    # Naive iid approximation; overlapping trades can violate independence.
    r_ic_se=round(float(1/np.sqrt(len(valid)-1)),6) if len(valid)>1 else None
    return {"n":len(valid),"r_ic_spearman":r_ic,
        "r_ic_standard_error_approx":r_ic_se,
        "r_ic_95_ci_approx":[round(r_ic-1.96*r_ic_se,6),round(r_ic+1.96*r_ic_se,6)] if r_ic is not None and r_ic_se is not None else None,
        "r_ic_uncertainty_method":"naive iid approximation; dependence ignored",
        "tercile_top_minus_bottom_realized_r":round(top-bottom,6),"top20_n":int(n20),
        "top20_realized_r":round(float(y[ix].mean()),6),
        "top20_mean_net_pct":round(float(np.mean([float(valid[i]["net_pct"]) for i in ix])),6),
        "top20_win_rate_pct":round(100*float(np.mean([float(valid[i]["net_pct"])>0 for i in ix])),3),
        "predicted_r_p10_p50_p90":[round(float(x),6) for x in np.percentile(p,[10,50,90])],
        "realized_r_mean":round(float(y.mean()),6)}

def _decorate_features(trade,frame,computed,breadth,market_gap):
    idx=frame.index.get_indexer([pd.Timestamp(trade["signal_date"])],method="pad")[0]
    if idx<0:return trade
    close=frame.Close.astype(float);vol=frame.Volume.astype(float)
    px=float(close.iloc[idx]);c20=float(close.iloc[max(0,idx-20)]);c60=float(close.iloc[max(0,idx-60)])
    # Model inputs and planned risk are frozen at signal close. Candidate
    # builders also retain next-open execution fields for realized fills; never
    # let those future prices leak into model features or the R denominator.
    stop=float(trade.get("stop",px));take=float(trade.get("take",px))
    planned_downside=max(0.0,100*(1-stop/px)+0.35)
    planned_upside=max(0.0,100*(take/px-1)-0.35)
    planned_rr=planned_upside/planned_downside if planned_downside>0 else 0.0
    returns=close.pct_change().iloc[max(0,idx-20):idx+1].dropna()
    dollar=px*float(vol.iloc[idx])
    row=computed.iloc[idx]
    trade.update({"signal_trend_200_pct":100*(px/float(row.MA200)-1) if pd.notna(row.MA200) and row.MA200 else 0,
        "signal_drawdown_20_pct":100*(px/float(close.iloc[max(0,idx-20):idx+1].max())-1),
        "signal_return_20d_pct":100*(px/c20-1) if c20 else 0,
        "signal_return_60d_pct":100*(px/c60-1) if c60 else 0,
        "signal_volatility_20d_pct":100*float(returns.std(ddof=0)) if len(returns) else 0,
        "signal_turnover_log10":float(np.log10(max(dollar,1))),
        "entry_net_rr":planned_rr,"entry_net_upside_pct":planned_upside,
        "entry_net_downside_pct":planned_downside,
        "market_breadth_pct":float(breadth.get(trade["signal_date"],50)),
        "market_ma200_gap_pct":float(market_gap.get(trade["signal_date"],0))})
    return trade

def _make_cohort_context(frames):
    close=pd.concat({k:v.Close.astype(float) for k,v in frames.items()},axis=1).sort_index()
    ret=close.pct_change()
    breadth=(ret>0).mean(axis=1)*100
    proxy=close.median(axis=1,skipna=True);ma=proxy.rolling(200,min_periods=200).mean()
    gap=(proxy/ma-1)*100
    return ({str(pd.Timestamp(k).date()):float(v) for k,v in breadth.dropna().items()},
            {str(pd.Timestamp(k).date()):float(v) for k,v in gap.dropna().items()})

def _select_r_policy(scored,thresholds,development_r_ic=None):
    dev=_period(scored,"development");table={}
    for name in thresholds:
        passed=[r for r in dev if r_accepts(r,name,thresholds)]
        selected=_resolve_nonoverlapping(passed)
        table[name]={"candidate_n":len(passed),
                     "acceptance_rate_pct":round(100*len(passed)/len(dev),2) if dev else None,
                     "selected":_trade_metrics(selected),"r_rank":_rank_metrics(selected)}
    # A positive mean return alone cannot promote a ranker with no development
    # ordering power. Keep the policy diagnostics, but fail closed before choice.
    if development_r_ic is not None and development_r_ic<=0:
        return None,table,"NO_DEVELOPMENT_EDGE"
    eligible=[(v["selected"].get("mean_net_pct"),k) for k,v in table.items()
              if v["selected"].get("trades",0)>=MIN_POLICY_TRADES and v["selected"].get("mean_net_pct") is not None
              and v["selected"]["mean_net_pct"]>MIN_DEVELOPMENT_EDGE]
    if eligible:return max(eligible,key=lambda p:(p[0],p[1]))[1],table,"DEVELOPMENT_EDGE_SELECTED"
    all_dev=_trade_metrics(dev)
    if all_dev["trades"]>=MIN_POLICY_TRADES and all_dev.get("mean_net_pct") is not None and all_dev["mean_net_pct"]<=0:
        return None,table,"NO_DEVELOPMENT_EDGE"
    return None,table,"INSUFFICIENT_DEVELOPMENT_SAMPLE"

def _r_policy_period(scored,policy,thresholds,period):
    pool=_period(scored,period)
    if not policy:return {"status":"NO_DEVELOPMENT_EDGE","candidate_n":len(pool),"accepted_n":0,"acceptance_rate_pct":0.0,"trades":_trade_metrics([])}
    passed=[r for r in pool if r_accepts(r,policy,thresholds)]
    chosen=_resolve_nonoverlapping(passed)
    return {"candidate_n":len(pool),"accepted_n":len(passed),
        "acceptance_rate_pct":round(100*len(passed)/len(pool),2) if pool else None,
        "trades":_trade_metrics(chosen),"rank_metrics":_rank_metrics(pool),
        "selected_rank_metrics":_rank_metrics(chosen)}

def _model_diagnostics(rows):
    return {"r_rank_metrics":_rank_metrics(rows),"auroc":_auc(rows),
            "brier":_brier(rows,"model_probability"),"ece":_calibration(rows).get("ece"),
            "probability_spread":_probability_spread(rows),"calibration":_calibration(rows)}

def run():
    started=datetime.now(timezone.utc)
    universe=_audit_universe()
    cohorts={"KOSPI":universe["KR"][:30],"KOSDAQ":universe["KQ"][:30],
             "TIGER_ETF":universe["KR"][-2:],"SOXL":universe["US"][-1:]}
    baseline_path=Path("data/cec6d80-baseline.json")
    archived_baseline=json.loads(baseline_path.read_text(encoding="utf-8")) if baseline_path.exists() else {}
    out={"created_utc":started.isoformat(),"status":"RESEARCH_ONLY_NOT_LIVE_APPROVED",
        "sample_start":SAMPLE_START,"sample_start_by_cohort":{},
        "windows":{"development_end":TRAIN_END,"validation_end":VALID_END,"diagnostic_end":HOLDOUT_END},
        "rank_target": "realized net return percent / entry net downside percent",
        "feature_timing":"all model features and planned payoff/risk fields use signal-day close; next-open values are outcome/execution only",
        "model_comparison":{},"r_rank_metrics":{},"acceptance_rate":{},
        "cec6d80_baseline_source":archived_baseline.get("source_run_id"),
        "portfolio":{"max_concurrent_positions":5,"weighting":"equal initial notional slots"},"cohorts":{}}
    integrity=True
    for cohort,items in cohorts.items():
        legacy_by_variant={s:[] for s in PARAMS};broad=[];frames={};computed_by={};instruments=[];errors=[]
        for i,(ticker,name,market,sector) in enumerate(items,1):
            try:
                minimum_bars=1 if ticker=="0262B0" else MIN_HISTORY_BARS
                prices,source=get_historical_prices(ticker,market,start_date=SAMPLE_START,min_bars=minimum_bars)
                first=str(pd.Timestamp(prices.index[0]).date());last=str(pd.Timestamp(prices.index[-1]).date())
                post_period_listing=first>HOLDOUT_END
                if len(prices)<MIN_HISTORY_BARS and not post_period_listing:raise ValueError(f"history {len(prices)} bars < {MIN_HISTORY_BARS}")
                if last<MIN_END_DATE:raise ValueError(f"OHLCV ends {last}, before required {MIN_END_DATE}")
                comp,variants=candidates_from_frame(prices,ticker,market)
                frames[ticker]=prices;computed_by[ticker]=comp
                for strategy,ts in variants.items():legacy_by_variant[strategy].extend(t for t in ts if t["signal_date"]>=SAMPLE_START)
                broad.extend(t for t in broad_candidates_from_computed(comp,ticker,market,end_date=HOLDOUT_END,cost_pct=.35) if t["signal_date"]>=SAMPLE_START)
                instruments.append({"ticker":ticker,"name":name,"source":source,"first_date":first,"last_date":last,"bars":len(prices),"variants":variants})
            except Exception as exc:errors.append({"ticker":ticker,"name":name,"error":str(exc)[:220]})
            print(f"AUDIT_PROGRESS {cohort} {i}/{len(items)} loaded={len(instruments)}",flush=True)
        breadth,gap=_make_cohort_context(frames)
        legacy_all=[]
        for strategy,ts in legacy_by_variant.items():
            for t in ts:
                frame=frames.get(t["ticker"]);comp=computed_by.get(t["ticker"])
                if frame is not None:legacy_all.append(_decorate_features({**t,"candidate_family":"legacy","candidate_variant":strategy,"strategy":strategy},frame,comp,breadth,gap))
        broad_all=[]
        for t in broad:
            frame=frames.get(t["ticker"]);comp=computed_by.get(t["ticker"])
            if frame is not None:broad_all.append(_decorate_features({**t,"candidate_family":"broad","candidate_variant":"broad_market_ml","strategy":"broad_market_ml"},frame,comp,breadth,gap))
        combined=legacy_all+broad_all
        # Exclude symbols whose development history has too few completed candidates,
        # while keeping them in OHLCV integrity accounting and the coverage denominator.
        symbol_dev={ticker:sum(t["ticker"]==ticker and t["entry_date"]<TRAIN_END and t["exit_date"]<TRAIN_END for t in combined) for ticker in frames}
        eligible={t for t,n in symbol_dev.items() if n>=20}
        eligible_rows=[t for t in combined if t["ticker"] in eligible]
        breadth_dates=[str(pd.Timestamp(x).date()) for x in breadth]
        actual_starts={r["ticker"]:r["first_date"] for r in instruments}
        out["sample_start_by_cohort"][cohort]={"requested_start":SAMPLE_START,"actual_first_dates":actual_starts,
            "earliest_actual":min(actual_starts.values()) if actual_starts else None,
            "development_candidate_n":sum(1 for t in combined if _period([t],"development")),
            "development_closed_sample_n":sum(1 for t in combined if _period([t],"development") and t.get("net_pct") is not None),
            "model_eligible_symbols":sorted(eligible),"model_excluded_symbols":{k:v for k,v in symbol_dev.items() if k not in eligible}}
        dev_variant={s:_trade_metrics(_period(ts,"development")) for s,ts in legacy_by_variant.items()}
        valid_variants=[s for s,m in dev_variant.items() if m["trades"]>=MIN_POLICY_TRADES and m.get("mean_net_pct") is not None]
        legacy_choice=max(valid_variants,key=lambda s:(dev_variant[s]["mean_net_pct"],s),default=None)
        legacy_selected=_resolve_nonoverlapping(_period(legacy_by_variant.get(legacy_choice,[]),"validation")+_period(legacy_by_variant.get(legacy_choice,[]),"diagnostic")) if legacy_choice else []
        scored_configs={}; statuses={}; dev_comp={}
        for config in MODEL_CONFIGS:
            min_train=MIN_TRAIN_SAMPLES_BY_COHORT[cohort]
            scored_configs[config],statuses[config]=walk_forward_scores(eligible_rows,config,min_train)
            dev=_period(scored_configs[config],"development")
            dev_comp[config]={"r_rank_metrics":_rank_metrics(dev),"auroc":_auc(dev),"candidate_n":len(dev),
                "minimum_train_samples":min_train,"max_training_samples":4000,"features":30}
        selectable=[(v["r_rank_metrics"].get("r_ic_spearman"),v["r_rank_metrics"].get("tercile_top_minus_bottom_realized_r"),k)
                    for k,v in dev_comp.items() if v["r_rank_metrics"].get("r_ic_spearman") is not None and v["candidate_n"]>=MIN_POLICY_TRADES]
        selected=max(selectable,key=lambda p:(p[0],p[1] if p[1] is not None else -1,p[2]))[2] if selectable else None
        scored=scored_configs.get(selected,[])
        thresholds=make_variant_thresholds(_period(scored,"development")) if selected else {}
        selected_dev_r_ic=dev_comp.get(selected,{}).get("r_rank_metrics",{}).get("r_ic_spearman") if selected else None
        policy,policy_dev,policy_status=(_select_r_policy(scored,thresholds,selected_dev_r_ic) if selected
            else (None,{},"INSUFFICIENT_DEVELOPMENT_RANKING_SAMPLE"))
        model_trades=_resolve_nonoverlapping([t for t in scored if t["ticker"] in eligible and policy and r_accepts(t,policy,thresholds)])
        cohort_result={"requested":len(items),"loaded":len(instruments),"errors":errors,
            "data_coverage_pct":round(100*len(instruments)/len(items),2) if items else None,
            "data_complete":len(instruments)==len(items) and not errors,
            "model_eligible_symbols_n":len(eligible),"model_coverage_of_requested_pct":round(100*len(eligible)/len(items),2) if items else None,
            "candidate_pool":{"legacy":len(legacy_all),"broad":len(broad_all),"eligible_combined":len(eligible_rows)},
            "legacy_selected_development_strategy":legacy_choice,"legacy_development_by_variant":dev_variant,
            "model_comparison":dev_comp,"selected_model":selected,"selected_hyperparameters":MODEL_CONFIGS.get(selected),
            "minimum_training_samples":MIN_TRAIN_SAMPLES_BY_COHORT[cohort],
            "model_selection_metric":"development prequential Spearman R-IC, tie-break by top-bottom tercile realized R spread",
            "policy_selection_rank_ic_gate":{"criterion":"selected development R-IC > 0","development_r_ic":selected_dev_r_ic,
                "passed":bool(selected_dev_r_ic is not None and selected_dev_r_ic>0)},
            "selected_model_policy":policy,
            "selected_model_status":policy_status,"development_predicted_r_thresholds":thresholds,
            "development_policy_comparison":policy_dev,"periods":{},"instruments":[]}
        for period in ("development","validation","diagnostic"):
            legacy_pool=_period(legacy_by_variant.get(legacy_choice,[]),period) if legacy_choice else []
            legacy_chosen=_resolve_nonoverlapping(legacy_pool)
            model_pool=_period(model_trades,period)
            all_scored=_period(scored,period)
            accepted_candidates=[r for r in all_scored if policy and r_accepts(r,policy,thresholds)]
            legacy_m=_metrics_with_portfolio(legacy_chosen,len(items),frames,period) if period!="development" else _trade_metrics(legacy_chosen)
            model_m=_metrics_with_portfolio(model_pool,len(items),frames,period) if period!="development" else _trade_metrics(model_pool)
            model_m.update(_coverage(model_pool,len(items)))
            model_m["acceptance_rate_pct"]=round(100*len(accepted_candidates)/len(all_scored),2) if all_scored else 0.0
            diag=_model_diagnostics(all_scored)
            cec6=None
            cohort_result["periods"][period]={"legacy":legacy_m,"new_model":model_m,
                "cec6d80_model":archived_baseline.get("cohorts",{}).get(cohort,{}).get("periods",{}).get(period,{}).get("model"),
                "cec6d80_legacy":archived_baseline.get("cohorts",{}).get(cohort,{}).get("periods",{}).get(period,{}).get("legacy"),
                "accepted_candidates_n":len(accepted_candidates),"scored_candidate_n":len(all_scored),
                "diagnostics":diag,"r_rank_metrics":_rank_metrics(all_scored),
                "unselected_diagnostic":{"legacy_candidate_n":len(_period(legacy_all,period)),"broad_candidate_n":len(_period(broad_all,period)),
                    "legacy_realized":_trade_metrics(_period(legacy_all,period)),"broad_realized":_trade_metrics(_period(broad_all,period))}}
        for inst in instruments:
            ticker=inst["ticker"]
            inst["legacy"]={period:_trade_metrics([t for t in _period(legacy_by_variant.get(legacy_choice,[]) if legacy_choice else [],period) if t["ticker"]==ticker]) for period in ("validation","diagnostic")}
            inst["model_eligible"] = ticker in eligible
            inst["development_candidate_n"]=symbol_dev.get(ticker,0)
            inst["model"]={period:_trade_metrics([t for t in model_trades if t["ticker"]==ticker and _period([t],period)]) for period in ("validation","diagnostic")}
            cohort_result["instruments"].append(inst)
        out["model_comparison"][cohort]=dev_comp
        out["r_rank_metrics"][cohort]={p:cohort_result["periods"][p]["r_rank_metrics"] for p in cohort_result["periods"]}
        out["acceptance_rate"][cohort]={p:cohort_result["periods"][p]["new_model"]["acceptance_rate_pct"] for p in cohort_result["periods"]}
        out["cohorts"][cohort]=cohort_result
        integrity &= cohort_result["data_complete"]
        print("AUDIT_RESULT "+json.dumps({"cohort":cohort,"loaded":len(instruments),"legacy":legacy_choice,
            "model":selected,"status":policy_status,"validation":cohort_result["periods"]["validation"],
            "diagnostic":cohort_result["periods"]["diagnostic"]},ensure_ascii=False),flush=True)
    out["data_integrity_status"]="PASS" if integrity else "FAIL"
    out["runtime_seconds"]=(datetime.now(timezone.utc)-started).total_seconds()
    Path("data").mkdir(exist_ok=True);Path("data/audit-63-latest.json").write_text(json.dumps(out,ensure_ascii=False,indent=2,default=str),encoding="utf-8")
    if not integrity:raise SystemExit("FAILED: incomplete 63-instrument OHLCV coverage; see data/audit-63-latest.json")
    return out

if __name__=="__main__":run()
