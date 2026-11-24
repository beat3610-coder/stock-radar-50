"""Leakage-controlled 63-instrument legacy-vs-model audit.

The holdout dates are fixed. All candidate families share one feature schema,
are scored prequentially, and model configuration/policy selection uses only
development observations. This file is research-only and never approves live use.
"""
import json
from collections import Counter
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from universe import get_universe, MANDATORY_KR, MANDATORY_US, AUDIT_63_BENCHMARK
from naver_api_universe import naver_api_top100
from providers import (get_historical_prices,get_market_series_bundle,
                       get_historical_investor_flows)
from strategy_lab import candidates_from_frame, TRAIN_END, VALID_END, HOLDOUT_END, PARAMS
from broad_candidates import broad_candidates_from_computed
from predictive_model import CONFIGS, DEFAULT_CONFIG, prepare_snapshot, predict_from_snapshot
from audit_300 import metrics
from market_context import build_market_context, REGIME_RULE, INDEX_FOR_COHORT
from disclosures import (get_cached_company_disclosures,request_interval_seconds,disclosure_features_for_signal,
                         BAD_KEYWORDS,BAD_TITLE_EXCLUSIONS,bad_disclosure_keyword_matches,
                         POSITIVE_KEYWORDS,PERIODIC_KEYWORDS)
from concurrent.futures import ThreadPoolExecutor, as_completed
from r_rank_model import (FEATURE_GROUPS, NUMERIC_FEATURES, CENTERS, SCALES, STRATEGIES,
                          MIN_TRAIN_SAMPLES_BY_COHORT, MODEL_CONFIGS, accepts as r_accepts,
                          make_variant_thresholds, walk_forward_scores)

SAMPLE_START="2024-07-01"
MIN_HISTORY_BARS=250
MIN_END_DATE=(date.fromisoformat(HOLDOUT_END)+timedelta(days=12)).isoformat()
MAX_PORTFOLIO_POSITIONS=5
PORTFOLIO_COST_PCT=0.35
MIN_POLICY_TRADES=10
MIN_DEVELOPMENT_EDGE=0.0
EV_QUANTILES=(0.70,0.80,0.90)
TREND_MARKET_MA200_GAPS=(-5.0,-3.0,0.0)
TREND_VOLATILITY_QUANTILES=(0.80,0.90)
DISCLOSURE_BAD_MAX_GRID=(0,1)
DISCLOSURE_PERIODIC_DAYS_FLOOR_GRID=(0,20,60)
FOCUSED_HISTORY_START="2015-12-01"
FOCUSED_FIRST_BAR_CUTOFF="2016-01-05"
FOCUSED_MIN_DATA_COMPLETENESS=0.95
FOCUSED_MIN_DEVELOPMENT_BARS=200
FOCUSED_UNIVERSE_SIZES=(10,20,30)
FOCUSED_CORE_KOSPI=("005930","000660","035420")

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
    if period=="development":start,end=SAMPLE_START,TRAIN_END
    elif period=="validation":start,end=TRAIN_END,VALID_END
    else:start,end=VALID_END,HOLDOUT_END
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

def _frozen_audit_cohorts(universe):
    """Resolve names/sectors from metadata while keeping the audited symbols fixed."""
    markets={"KOSPI":("KR","KR"),"KOSDAQ":("KQ","KQ")}
    result={}
    for cohort,(key,market) in markets.items():
        by_code={str(item[0]):item for item in universe.get(key,[])}
        rows=[]
        for code in AUDIT_63_BENCHMARK[cohort]:
            item=by_code.get(code)
            rows.append(item if item is not None else (code,code,market,"sector metadata unavailable"))
        result[cohort]=rows
    kr_by_code={str(item[0]):item for item in universe.get("KR",[])}
    etfs=[]
    for code,name,_ in MANDATORY_KR:
        item=kr_by_code.get(code)
        etfs.append(item if item is not None and len(item)>=4 else (code,name,"KR","ETF"))
    result["TIGER_ETF"]=etfs
    result["SOXL"]=[("SOXL","SOXL","US","leveraged ETF")]
    return result

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
    cohorts=_frozen_audit_cohorts(universe)
    out={"created_utc":datetime.now(timezone.utc).isoformat(),"status":"RESEARCH_ONLY_NOT_LIVE_APPROVED",
         "sample_start":SAMPLE_START,"windows":{"development_end":TRAIN_END,"validation_end":VALID_END,"diagnostic_end":HOLDOUT_END},
         "cost_round_trip_pct":0.35,"portfolio":{"max_concurrent_positions":MAX_PORTFOLIO_POSITIONS,
         "weighting":"equal initial notional slots","mdd":"daily mark-to-market"},
         "universe_sources":{k:universe.get(k+"_source") for k in ("KR","KQ","US")},
         "frozen_audit_benchmark":{k:list(v) for k,v in AUDIT_63_BENCHMARK.items()},"cohorts":{}}
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
    by_day={}
    for row in valid:by_day.setdefault(row.get("signal_date"),[]).append(row)
    eligible_days=[];excluded_days=0
    for day,group in by_day.items():
        if len(group)<3:
            excluded_days+=1;continue
        value=spearmanr([float(x["model_predicted_r"]) for x in group],
                        [float(x["realized_r"]) for x in group]).statistic
        if value is not None and np.isfinite(value):eligible_days.append(float(value))
    date_mean=float(np.mean(eligible_days)) if eligible_days else None
    date_se=float(np.std(eligible_days,ddof=1)/np.sqrt(len(eligible_days))) if len(eligible_days)>1 else None
    return {"n":len(valid),"r_ic_spearman":r_ic,
        "datewise_spearman_mean":round(date_mean,6) if date_mean is not None else None,
        "datewise_spearman_se":round(date_se,6) if date_se is not None else None,
        "datewise_scored_days_n":len(eligible_days),"datewise_excluded_days_under_3_candidates":excluded_days,
        "r_ic_standard_error_approx":r_ic_se,
        "r_ic_95_ci_approx":[round(r_ic-1.96*r_ic_se,6),round(r_ic+1.96*r_ic_se,6)] if r_ic is not None and r_ic_se is not None else None,
        "r_ic_uncertainty_method":"naive iid approximation; dependence ignored",
        "tercile_top_minus_bottom_realized_r":round(top-bottom,6),"top20_n":int(n20),
        "top20_realized_r":round(float(y[ix].mean()),6),
        "top20_mean_net_pct":round(float(np.mean([float(valid[i]["net_pct"]) for i in ix])),6),
        "top20_win_rate_pct":round(100*float(np.mean([float(valid[i]["net_pct"])>0 for i in ix])),3),
        "predicted_r_p10_p50_p90":[round(float(x),6) for x in np.percentile(p,[10,50,90])],
        "realized_r_mean":round(float(y.mean()),6)}

def _feature_coverage(rows):
    """Development-only availability/variation audit; returns constant columns to drop."""
    report={};excluded=[]
    columns={key:[] for key in NUMERIC_FEATURES}
    for strategy in STRATEGIES:
        columns[f"strategy_flag:{strategy}"]=[]
        columns[f"strategy_x_entry_net_rr:{strategy}"]=[]
    for row in rows:
        strategy=row.get("strategy") or row.get("candidate_variant")
        if strategy not in STRATEGIES and row.get("candidate_family")=="broad":strategy="broad_market_ml"
        for candidate in STRATEGIES:
            flag=1.0 if strategy==candidate else 0.0
            columns[f"strategy_flag:{candidate}"].append(flag)
            try:rr=float(row["entry_net_rr"])
            except (KeyError,TypeError,ValueError):rr=CENTERS["entry_net_rr"]
            if not np.isfinite(rr):rr=CENTERS["entry_net_rr"]
            columns[f"strategy_x_entry_net_rr:{candidate}"].append(
                flag*np.clip((rr-CENTERS["entry_net_rr"])/SCALES["entry_net_rr"],-4,4))
    for key in NUMERIC_FEATURES:
        raw=[]
        for row in rows:
            try:value=float(row[key])
            except (KeyError,TypeError,ValueError):value=np.nan
            raw.append(value if np.isfinite(value) else np.nan)
        columns[key]=raw
    for key,raw in columns.items():
        finite=np.asarray([x for x in raw if np.isfinite(x)],dtype=float)
        filled=np.asarray([x if np.isfinite(x) else CENTERS[key] for x in raw],dtype=float)
        std=float(np.std(filled)) if len(filled) else 0.0
        unique=int(np.unique(filled).size) if len(filled) else 0
        is_constant=unique<=1 or std<=1e-12
        report[key]={"rows_n":len(rows),"non_missing_n":int(len(finite)),
            "non_missing_pct":round(100*len(finite)/len(rows),2) if rows else 0.0,
            "unique_values_n":unique,"standard_deviation":round(std,10),
            "auto_excluded_constant":bool(is_constant),
            "missing_imputation":"fixed domain center"}
        if is_constant:excluded.append(key)
    return report,excluded

def _datewise_thresholds(dev_scored):
    eligible,excluded_days=_datewise_policy_rows(dev_scored)
    values=np.asarray([float(x["model_predicted_r"]) for x in eligible],dtype=float)
    out={"excluded_development_days_under_3_candidates":excluded_days,"eligible_development_rows_n":len(eligible)}
    for q in (.70,.80,.90):
        name=f"r_p{int(q*100)}"
        out[name]={"global":float(np.quantile(values,q)) if len(values) else None,
                   "quantile":q,"eligible_rows_n":len(values)}
    return out

def _datewise_policy_rows(rows):
    counts={}
    for row in rows:counts[row.get("signal_date")]=counts.get(row.get("signal_date"),0)+1
    eligible=[r for r in rows if counts.get(r.get("signal_date"),0)>=3]
    return eligible,sum(n<3 for n in counts.values())

def _datewise_accepts(row,policy,thresholds,period_rows):
    spec=thresholds.get(policy,{})
    cutoff=spec.get("global")
    if cutoff is None:return False
    eligible,_=_datewise_policy_rows(period_rows)
    allowed={(r.get("ticker"),r.get("signal_date"),r.get("entry_date"),r.get("strategy")) for r in eligible}
    key=(row.get("ticker"),row.get("signal_date"),row.get("entry_date"),row.get("strategy"))
    return key in allowed and float(row.get("model_predicted_r",-np.inf))>=float(cutoff)

def _datewise_selected(rows,policy,thresholds):
    eligible,_=_datewise_policy_rows(rows)
    cutoff=thresholds.get(policy,{}).get("global")
    if cutoff is None:return []
    return [r for r in eligible if float(r.get("model_predicted_r",-np.inf))>=float(cutoff)]

def _target_variant_choices(results):
    choices=[]
    for name,data in results.items():
        metrics=data.get("selected_dev_metrics") or {}
        primary=metrics.get("datewise_spearman_mean")
        if primary is not None:choices.append((primary,name))
    return choices

def _legacy_policy_rows(legacy_rows,scored_rows,kind,threshold=None):
    key=lambda r:(r.get("ticker"),r.get("signal_date"),r.get("entry_date"),r.get("strategy"))
    score_by_key={key(r):r for r in scored_rows}
    eligible_scored,_=_datewise_policy_rows(scored_rows)
    eligible_days={r.get("signal_date") for r in eligible_scored}
    selected=[]
    for row in legacy_rows:
        if kind=="legacy_only":selected.append(row);continue
        if kind in ("legacy_regime_gate","legacy_model_filter_and_regime_gate"):
            if row.get("market_regime")=="RISK_OFF":continue
            score=score_by_key.get(key(row))
            if kind=="legacy_model_filter_and_regime_gate":
                if score is None or row.get("signal_date") not in eligible_days:continue
                if threshold is None or float(score.get("model_predicted_r",-np.inf))<float(threshold):continue
            selected.append({**row,**({"model_predicted_r":score.get("model_predicted_r"),"realized_r":score.get("realized_r")} if score else {})})
            continue
        score=score_by_key.get(key(row))
        if score is None or row.get("signal_date") not in eligible_days:continue
        if float(score.get("model_predicted_r",-np.inf))>=float(threshold):
            selected.append({**row,"model_predicted_r":score["model_predicted_r"],
                             "realized_r":score.get("realized_r")})
    return _resolve_nonoverlapping(selected)


def _risk_off_blocked(trade):
    """The frozen market-context rule blocks every new entry in RISK_OFF."""
    return trade.get("market_regime")=="RISK_OFF"


def _legacy_policy_stage(rows,stage,params=None):
    """Apply one legacy policy-ladder stage using only signal-time fields."""
    params=params or {}
    if stage=="legacy_only":return _resolve_nonoverlapping(rows)
    selected=[]
    for row in rows:
        if _risk_off_blocked(row):continue
        if stage=="legacy_risk_off_hard_gate":
            selected.append(row);continue
        try:
            market_gap=float(row["market_ma200_gap_pct"])
            stock_trend=float(row["signal_trend_200_pct"])
            excess60=float(row["relative_strength_60d_pct"])
            volatility=float(row["signal_volatility_20d_pct"])
            if not all(np.isfinite(x) for x in (market_gap,stock_trend,excess60,volatility)):continue
        except (KeyError,TypeError,ValueError):continue
        if (market_gap<float(params.get("market_ma200_gap_min_pct",np.inf)) or stock_trend<=0
                or excess60<=0 or volatility>float(params.get("volatility_cap_pct",-np.inf))):continue
        if stage in ("legacy_risk_off_trend_disclosure","legacy_risk_off_trend_disclosure_short_fundamental"):
            try:
                bad=float(row["disclosure_bad_60d"]);periodic=float(row["disclosure_periodic_days_since"])
                if not np.isfinite(bad) or not np.isfinite(periodic):continue
            except (KeyError,TypeError,ValueError):continue
            if bad>float(params.get("disclosure_bad_60d_max",0)):continue
            if periodic<float(params.get("disclosure_periodic_days_since_floor",0)):continue
        selected.append(row)
    return _resolve_nonoverlapping(selected)


def _disclosure_gate_input_coverage(rows,trend_params=None):
    """Count D-stage rows omitted solely because required disclosure fields are missing."""
    if trend_params:
        upstream=_legacy_policy_stage(rows,"legacy_risk_off_trend_filter",trend_params)
        basis="after selected C-stage RISK_OFF/trend filters"
    else:
        upstream=_legacy_policy_stage(rows,"legacy_risk_off_hard_gate",{})
        basis="after B-stage RISK_OFF filter; no selected C parameters"
    usable=[]
    for row in upstream:
        try:
            bad=float(row.get("disclosure_bad_60d",np.nan))
            periodic=float(row.get("disclosure_periodic_days_since",np.nan))
        except (TypeError,ValueError):continue
        if np.isfinite(bad) and np.isfinite(periodic):usable.append(row)
    denominator=len(upstream)
    return {"candidate_rows_before_disclosure_gate_n":denominator,
        "rows_with_disclosure_data_n":len(usable),
        "excluded_missing_disclosure_n":denominator-len(usable),
        "disclosure_coverage_pct":round(100*len(usable)/denominator,2) if denominator else 0.0,
        "disclosure_coverage_applicable":bool(denominator),
        "disclosure_coverage_basis":basis}


def _not_evaluated_fundamental_stage():
    return {"status":"NOT_EVALUATED","periods":{},"reason":{
        "short_selling":"Unavailable from unauthenticated CI sources: KRX endpoints return HTTP 400 LOGOUT; Naver status uses the same blocked KRX iframe.",
        "earnings_numeric_series":"Insufficient history: Naver has 3 annual years and 5 quarters; Yahoo has 4 annual years and 5 quarters, not the 2016-2025 development window.",
        "opendart":"API key required; no key or fabricated zero values used."}}


POLICY_SELECTION_METRIC="development mean_net_pct / max(abs(sequence MDD),1)"


def _risk_adjusted_policy_selection(steps):
    """Select A-D using positive development return scaled by sequence MDD."""
    comparison={};eligible=[]
    stage_names=("legacy_only","legacy_risk_off_hard_gate","legacy_risk_off_trend_filter",
                 "legacy_risk_off_trend_disclosure")
    for name in stage_names:
        step=steps.get(name,{})
        dev=step.get("periods",{}).get("development",{})
        metric=dev.get("selected_trades") or dev
        try:n=int(metric.get("trades",0))
        except (TypeError,ValueError):n=0
        try:mean=float(metric.get("mean_net_pct"))
        except (TypeError,ValueError):mean=float("nan")
        try:mdd=float(metric.get("max_drawdown_sequence_pct"))
        except (TypeError,ValueError):mdd=float("nan")
        score=(mean/max(abs(mdd),1.0) if np.isfinite(mean) and np.isfinite(mdd) else None)
        is_candidate=(step.get("status")=="EVALUATED" and n>=MIN_POLICY_TRADES
                      and np.isfinite(mean) and mean>0 and score is not None)
        if is_candidate:eligible.append((score,abs(mdd),name))
        if not is_candidate:
            if step.get("status")!="EVALUATED":reason=step.get("status","NOT_EVALUATED")
            elif n<MIN_POLICY_TRADES:reason="BELOW_MIN_POLICY_TRADES"
            elif not np.isfinite(mean) or mean<=0:reason="NONPOSITIVE_OR_MISSING_DEVELOPMENT_MEAN_NET"
            else:reason="MISSING_DEVELOPMENT_SEQUENCE_MDD"
        else:reason="ELIGIBLE"
        step["development_risk_adjusted_score"]=round(float(score),10) if score is not None else None
        comparison[name]={"status":step.get("status"),"development_trades":n,
            "development_mean_net_pct":round(mean,8) if np.isfinite(mean) else None,
            "development_sequence_mdd_pct":round(mdd,8) if np.isfinite(mdd) else None,
            "development_5slot_portfolio_mdd_pct":metric.get("max_drawdown_portfolio_5_equal_pct"),
            "development_risk_adjusted_score":step["development_risk_adjusted_score"],
            "candidate":is_candidate,"exclusion_reason":reason}
    selected=(sorted(eligible,key=lambda row:(-row[0],row[1],row[2]))[0][2]
              if eligible else "legacy_only")
    return selected,{"selection_metric":POLICY_SELECTION_METRIC,"stages":comparison,
        "candidate_requirements":{"development_trades_min":MIN_POLICY_TRADES,
            "development_mean_net_pct_gt":0},
        "tie_break_order":["higher risk-adjusted score","lower absolute sequence MDD","stage name ascending"],
        "fallback":"legacy_only","selected_policy":selected}


def _policy_ladder(legacy_rows,frames,requested,cohort,disclosures_available):
    """Score A-D on all fixed periods; select parameters and policy on development only."""
    period_names=("development","validation","diagnostic")
    dev_a=_resolve_nonoverlapping(_period(legacy_rows,"development"))
    base_dev=_trade_metrics(dev_a)
    vol_values=[]
    for row in dev_a:
        try:
            value=float(row["signal_volatility_20d_pct"])
            if np.isfinite(value):vol_values.append(value)
        except (KeyError,TypeError,ValueError):pass
    vol_caps={f"p{int(q*100)}":float(np.quantile(vol_values,q)) for q in TREND_VOLATILITY_QUANTILES} if vol_values else {}
    steps={}
    if isinstance(disclosures_available,dict):
        disclosure_available_symbols=int(disclosures_available.get("available_symbols_n",0))
        disclosure_requested_symbols=int(disclosures_available.get("requested_symbols_n",requested))
        symbol_coverage_pct=float(disclosures_available.get("coverage_pct",0.0))
    else:
        disclosure_available_symbols=1 if disclosures_available else 0
        disclosure_requested_symbols=max(1,requested)
        symbol_coverage_pct=100.0 if disclosures_available else 0.0
    def stage_periods(stage,params=None):
        report={}
        for period in period_names:
            source_rows=_period(legacy_rows,period)
            rows=_legacy_policy_stage(source_rows,stage,params)
            report[period]=_metrics_with_portfolio(rows,requested,frames,period)
            report[period]["small_sample"]=period=="validation" and report[period].get("trades",0)<MIN_POLICY_TRADES
            if stage=="legacy_risk_off_trend_disclosure":
                report[period].update(_disclosure_gate_input_coverage(source_rows,params))
        return report
    steps["legacy_only"]={"status":"EVALUATED","params":{},"periods":stage_periods("legacy_only")}
    steps["legacy_risk_off_hard_gate"]={"status":"EVALUATED","params":{"rule":REGIME_RULE["risk_off"]},
        "periods":stage_periods("legacy_risk_off_hard_gate")}

    trend_grid=[]
    for gap_min in TREND_MARKET_MA200_GAPS:
        for vol_name,vol_cap in vol_caps.items():
            params={"market_ma200_gap_min_pct":gap_min,"stock_ma200_gap_gt_pct":0,
                "relative_strength_60d_gt_pct":0,"volatility_cap_quantile":vol_name,
                "volatility_cap_pct":vol_cap}
            dev_rows=_legacy_policy_stage(dev_a,"legacy_risk_off_trend_filter",params)
            met=_trade_metrics(dev_rows)
            trend_grid.append({"params":params,"development":met})
    valid_trend=[x for x in trend_grid if x["development"].get("trades",0)>=MIN_POLICY_TRADES]
    selected_trend=max(valid_trend,key=lambda x:(x["development"].get("mean_net_pct",-np.inf),-x["development"].get("trades",0))) if valid_trend else None
    if selected_trend:
        steps["legacy_risk_off_trend_filter"]={"status":"EVALUATED","params":selected_trend["params"],
            "selection_rule":"development net per trade maximized among predeclared settings with at least MIN_POLICY_TRADES",
            "development_grid":trend_grid,"periods":stage_periods("legacy_risk_off_trend_filter",selected_trend["params"])}
    else:
        steps["legacy_risk_off_trend_filter"]={"status":"INSUFFICIENT_DEVELOPMENT_SAMPLE","params":None,
            "development_grid":trend_grid,"periods":{p:_metrics_with_portfolio([],requested,frames,p) for p in period_names}}

    dgrid=[]
    disclosure_params_for_coverage=selected_trend["params"] if selected_trend else None
    dev_disclosure_coverage=_disclosure_gate_input_coverage(dev_a,disclosure_params_for_coverage)
    row_coverage_low=(dev_disclosure_coverage["disclosure_coverage_applicable"] and
        dev_disclosure_coverage["disclosure_coverage_pct"]<50.0)
    partial_coverage=(symbol_coverage_pct<50.0 or row_coverage_low)
    def empty_disclosure_periods():
        periods={}
        for period in period_names:
            periods[period]=_metrics_with_portfolio([],requested,frames,period)
            periods[period].update(_disclosure_gate_input_coverage(
                _period(legacy_rows,period),disclosure_params_for_coverage))
            periods[period]["small_sample"]=period=="validation" and periods[period].get("trades",0)<MIN_POLICY_TRADES
        return periods
    if disclosure_available_symbols>0:
        for trend in trend_grid:
            for bad_max in DISCLOSURE_BAD_MAX_GRID:
                for periodic_floor in DISCLOSURE_PERIODIC_DAYS_FLOOR_GRID:
                    params={**trend["params"],"disclosure_bad_60d_max":bad_max,
                        "disclosure_periodic_days_since_floor":periodic_floor}
                    dev_rows=_legacy_policy_stage(dev_a,"legacy_risk_off_trend_disclosure",params)
                    dgrid.append({"params":params,"development":_trade_metrics(dev_rows)})
        valid_d=[x for x in dgrid if x["development"].get("trades",0)>=MIN_POLICY_TRADES]
        selected_d=max(valid_d,key=lambda x:(x["development"].get("mean_net_pct",-np.inf),-x["development"].get("trades",0))) if valid_d else None
        if selected_d:
            steps["legacy_risk_off_trend_disclosure"]={
                "status":"INSUFFICIENT_DEVELOPMENT_SAMPLE" if partial_coverage else "EVALUATED",
                "params":None if partial_coverage else selected_d["params"],
                "selection_rule":"development net per trade maximized among predeclared settings with at least MIN_POLICY_TRADES",
                "development_grid":dgrid,
                "periods":empty_disclosure_periods() if partial_coverage else
                    stage_periods("legacy_risk_off_trend_disclosure",selected_d["params"])}
        else:
            steps["legacy_risk_off_trend_disclosure"]={
                "status":"INSUFFICIENT_DEVELOPMENT_SAMPLE",
                "params":None,"development_grid":dgrid,"periods":empty_disclosure_periods()}
    else:
        steps["legacy_risk_off_trend_disclosure"]={"status":"NOT_EVALUATED","params":None,
            "development_grid":[],"periods":empty_disclosure_periods(),
            "reason":"No symbol met the 80% requested-year availability rule; missing disclosure fields remain NaN and do not add trades."}
    d_step=steps["legacy_risk_off_trend_disclosure"]
    d_step["disclosure_coverage"]={"available_symbols_n":disclosure_available_symbols,
        "requested_symbols_n":disclosure_requested_symbols,"symbol_coverage_pct":round(symbol_coverage_pct,2),
        "minimum_coverage_for_full_evaluation_pct":50.0,
        "development_candidate_rows":dev_disclosure_coverage,
        "coverage_limited":bool(partial_coverage),
        "status_reason":"Available coverage below 50%; D is not promoted and no trades are added." if partial_coverage else None,
        "selection_eligible":d_step.get("status")=="EVALUATED"}
    steps["legacy_risk_off_trend_disclosure_short_fundamental"]=_not_evaluated_fundamental_stage()

    selected,selection_comparison=_risk_adjusted_policy_selection(steps)
    steps["selected_policy"]=selected
    steps["selection_metric"]=POLICY_SELECTION_METRIC
    steps["development_selection_comparison"]=selection_comparison
    steps["selection_rule"]=("Among A-D, select the development candidate with the highest mean_net_pct / "
        "max(abs(sequence MDD),1), requiring at least MIN_POLICY_TRADES and positive development mean net; "
        "ties use lower absolute sequence MDD then stage name ascending. A is scored on the same rule and is "
        "the fallback when no candidate qualifies. Validation and diagnostic are report-only; E is NOT_EVALUATED.")
    steps["development_volatility_caps_pct"]={k:round(v,8) for k,v in vol_caps.items()}
    steps["grid_constants"]={"market_ma200_gap_min_pct":list(TREND_MARKET_MA200_GAPS),
        "stock_ma200_gap_gt_pct":0,"relative_strength_60d_gt_pct":0,
        "volatility_quantiles":list(TREND_VOLATILITY_QUANTILES),
        "disclosure_bad_60d_max":list(DISCLOSURE_BAD_MAX_GRID),
        "disclosure_periodic_days_since_floor":list(DISCLOSURE_PERIODIC_DAYS_FLOOR_GRID),
        "min_policy_trades":MIN_POLICY_TRADES,"risk_off_rule":REGIME_RULE["risk_off"]}
    return steps

def _policy_candidate_metrics(legacy_by_period,scored_by_period,frames,requested,period,thresholds):
    specs={"legacy_only":{"kind":"legacy_only"}}
    specs.update({f"legacy_model_filter_{name}":{"kind":"model_filter","threshold":row.get("global")}
                  for name,row in thresholds.items() if name.startswith("r_p")})
    specs.update({f"legacy_model_filter_{name}_and_risk_off_gate":{"kind":"legacy_model_filter_and_regime_gate","threshold":row.get("global")}
                  for name,row in thresholds.items() if name.startswith("r_p")})
    specs["legacy_regime_gate"]={"kind":"legacy_regime_gate"}
    result={}
    for name,spec in specs.items():
        if spec["kind"]=="legacy_only":rows=_legacy_policy_rows(legacy_by_period,scored_by_period,"legacy_only")
        elif spec["kind"]=="legacy_regime_gate":rows=_legacy_policy_rows(legacy_by_period,scored_by_period,"legacy_regime_gate")
        elif spec["kind"]=="legacy_model_filter_and_regime_gate":rows=_legacy_policy_rows(legacy_by_period,scored_by_period,spec["kind"],spec.get("threshold"))
        elif spec.get("threshold") is not None:
            rows=_legacy_policy_rows(legacy_by_period,scored_by_period,"model_filter",spec["threshold"])
        else:rows=[]
        metric=_metrics_with_portfolio(rows,requested,frames,period)
        metric["scored_legacy_candidates_n"]=len(rows) if spec["kind"]!="legacy_only" else None
        scored_rows=[r for r in rows if "model_predicted_r" in r]
        result[name]={"threshold_predicted_y_cs":spec.get("threshold"),"trades":metric,
                      "datewise_rank_metrics":_rank_metrics(scored_rows) if scored_rows else None}
    return result

def _decorate_features(trade,frame,computed,breadth,market_gap,market_return60=None):
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
        "market_ma200_gap_pct":float(market_gap.get(trade["signal_date"],np.nan)),
        "market_return_60d_pct":float((market_return60 or {}).get(trade["signal_date"],np.nan)),
        "relative_strength_60d_pct":(100*(px/c60-1)-float(market_return60[trade["signal_date"]])
             if c60 and market_return60 is not None and trade["signal_date"] in market_return60 else np.nan)})
    return trade

def _make_cohort_context(frames):
    close=pd.concat({k:v.Close.astype(float) for k,v in frames.items()},axis=1).sort_index()
    ret=close.pct_change()
    breadth=(ret>0).mean(axis=1)*100
    proxy=close.median(axis=1,skipna=True);ma=proxy.rolling(200,min_periods=200).mean()
    gap=(proxy/ma-1)*100
    market_ret60=proxy.pct_change(60)*100
    return ({str(pd.Timestamp(k).date()):float(v) for k,v in breadth.dropna().items()},
            {str(pd.Timestamp(k).date()):float(v) for k,v in gap.dropna().items()},
            {str(pd.Timestamp(k).date()):float(v) for k,v in market_ret60.dropna().items()})

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

def _broad_risk_off_blocked(trade):
    """Frozen RISK_OFF hard gate for every new entry, including legacy candidates."""
    return trade.get("market_regime")=="RISK_OFF"

def _period_regime_counts(candidates,context,period):
    if period=="development":lo,hi=SAMPLE_START,TRAIN_END
    elif period=="validation":lo,hi=TRAIN_END,VALID_END
    else:lo,hi=VALID_END,HOLDOUT_END
    days={d:r for d,r in context["regime_by_date"].items() if lo<=d<hi}
    rows=[r for r in candidates if lo<=r.get("signal_date","")<hi]
    return {state:{"market_days":sum(v==state for v in days.values()),
        "candidate_n":sum(r.get("market_regime")==state for r in rows),
        "broad_candidate_n":sum(r.get("strategy")=="broad_market_ml" and r.get("market_regime")==state for r in rows),
        "legacy_candidate_n":sum(r.get("strategy")!="broad_market_ml" and r.get("market_regime")==state for r in rows)}
        for state in ("RISK_ON","NEUTRAL","RISK_OFF","UNKNOWN")}

def _regime_trade_metrics(trades):
    out={}
    for regime in ("RISK_ON","NEUTRAL","RISK_OFF","UNKNOWN"):
        group=[t for t in trades if t.get("market_regime")==regime]
        out[regime]=_trade_metrics(group)
    return out

def _disclosure_bad_title_counts(items,disclosure_by_ticker,limit=10):
    counts=Counter();matched_by_title={};seen=set()
    for item in items:
        ticker=str(item[0])
        if ticker in seen:continue
        seen.add(ticker)
        for row in disclosure_by_ticker.get(ticker,{}).get("records",[]):
            title=str(row.get("title", "")).strip()
            matches=bad_disclosure_keyword_matches(title)
            if matches:
                counts[title]+=1
                matched_by_title[title]=list(matches)
    return [{"title":title,"count":count,"matched_keywords":matched_by_title[title]}
        for title,count in sorted(counts.items(),key=lambda item:(-item[1],item[0]))[:limit]]


def _disclosure_failure_reasons_top(disclosure_by_ticker,limit=5):
    counts=Counter()
    for result in disclosure_by_ticker.values():
        for failure in result.get("diagnostics",{}).get("failures",[]):
            message=str(failure)
            status=re.search(r"HTTP(?:Error)?:\s*HTTP\s*(\d{3})",message,re.I)
            if status:reason=f"HTTP {status.group(1)}"
            elif "SSLError" in message:reason="SSLError"
            elif "ConnectionError" in message or "RemoteDisconnected" in message:reason="ConnectionError/RemoteDisconnected"
            elif "Timeout" in message:reason="Timeout"
            elif "duplicate page/no pagination progress" in message:reason="Duplicate page/no pagination progress"
            elif "reached page cap" in message:reason="Page cap reached"
            else:reason=message.split(":",1)[0][:100] or "Unknown failure"
            counts[reason]+=1
    return [{"reason":reason,"count":count}
        for reason,count in sorted(counts.items(),key=lambda item:(-item[1],item[0]))[:limit]]


def _focused_liquidity_ranking(cohort,items,frames):
    start=pd.Timestamp(FOCUSED_HISTORY_START);end=pd.Timestamp(TRAIN_END)
    expected_dates=set()
    normalized_indexes={}
    for ticker,frame in frames.items():
        idx=pd.DatetimeIndex(pd.to_datetime(frame.index))
        if idx.tz is not None:idx=idx.tz_localize(None)
        idx=idx.normalize();normalized_indexes[str(ticker)]=idx
        expected_dates.update(idx[(idx>=start)&(idx<end)].tolist())
    expected_n=len(expected_dates)
    rows=[]
    for ticker,name,market,sector in items:
        ticker=str(ticker);frame=frames.get(ticker);reasons=[]
        if frame is None:
            rows.append({"ticker":ticker,"name":name,"median_daily_turnover":None,
                "development_bars_n":0,"expected_development_sessions_n":expected_n,
                "development_bar_completeness_pct":0.0,"first_development_bar":None,
                "eligible":False,"liquidity_rank":None,"exclusion_reasons":["OHLCV_UNAVAILABLE"]})
            continue
        idx=normalized_indexes[ticker]
        mask=(idx>=start)&(idx<end)
        dev=frame.iloc[np.flatnonzero(mask)]
        dev_n=int(idx[mask].nunique())
        completeness=100.0*dev_n/expected_n if expected_n else 0.0
        first_bar=str(idx[mask].min().date()) if dev_n else None
        if first_bar is None:reasons.append("NO_DEVELOPMENT_BARS")
        elif pd.Timestamp(first_bar)>=pd.Timestamp(FOCUSED_FIRST_BAR_CUTOFF):
            reasons.append(f"FIRST_BAR_NOT_BEFORE_{FOCUSED_FIRST_BAR_CUTOFF}")
        if completeness/100.0<FOCUSED_MIN_DATA_COMPLETENESS:
            reasons.append("DEVELOPMENT_BAR_COMPLETENESS_BELOW_95_PERCENT")
        if dev_n<FOCUSED_MIN_DEVELOPMENT_BARS:
            reasons.append("INSUFFICIENT_DEVELOPMENT_LISTING_HISTORY")
        try:
            close=pd.to_numeric(dev["Close"],errors="coerce").to_numpy(dtype=float)
            volume=pd.to_numeric(dev["Volume"],errors="coerce").to_numpy(dtype=float)
            turnover=close*volume
            valid=turnover[np.isfinite(turnover)&(turnover>0)]
            median=float(np.median(valid)) if len(valid) else None
        except (KeyError,TypeError,ValueError):median=None
        if median is None:reasons.append("NO_VALID_DEVELOPMENT_TURNOVER")
        rows.append({"ticker":ticker,"name":name,"market":market,"sector":sector,
            "median_daily_turnover":round(median,2) if median is not None else None,
            "development_bars_n":dev_n,"expected_development_sessions_n":expected_n,
            "development_bar_completeness_pct":round(completeness,4),
            "first_development_bar":first_bar,"eligible":not reasons,
            "liquidity_rank":None,"exclusion_reasons":reasons})
    eligible=sorted((r for r in rows if r["eligible"]),
        key=lambda r:(-r["median_daily_turnover"],r["ticker"]))
    for rank,row in enumerate(eligible,1):row["liquidity_rank"]=rank
    return {"expected_development_sessions_n":expected_n,
        "expected_session_source":"union of the already-loaded frozen-cohort OHLCV dates in the development window",
        "selection_window_start":FOCUSED_HISTORY_START,"selection_window_end_exclusive":TRAIN_END,
        "minimum_data_completeness_pct":FOCUSED_MIN_DATA_COMPLETENESS*100,
        "first_bar_must_precede":FOCUSED_FIRST_BAR_CUTOFF,
        "minimum_development_bars":FOCUSED_MIN_DEVELOPMENT_BARS,
        "eligible_tickers_by_liquidity":[r["ticker"] for r in eligible],
        "symbols":sorted(rows,key=lambda r:(r["liquidity_rank"] is None,
                                              r["liquidity_rank"] or 10**9,r["ticker"]))}


def _focused_recommendation(sets,baseline_dev):
    def values(metric):
        try:n=int(metric.get("trades",0))
        except (TypeError,ValueError):n=0
        try:mean=float(metric.get("mean_net_pct"))
        except (TypeError,ValueError):mean=float("nan")
        try:mdd=float(metric.get("max_drawdown_sequence_pct"))
        except (TypeError,ValueError):mdd=float("nan")
        score=mean/abs(mdd) if np.isfinite(mean) and np.isfinite(mdd) and abs(mdd)>1e-12 else None
        return n,mean,mdd,score
    base_n,base_mean,base_mdd,base_score=values(baseline_dev)
    candidates=[]
    table=[]
    for name,row in sets.items():
        if not row.get("recommendation_candidate"):continue
        dev=row.get("periods",{}).get("development",{})
        n,mean,mdd,score=values(dev)
        qualifies=(n>=30 and np.isfinite(mean) and mean>0 and score is not None
                   and base_score is not None and score>base_score)
        reasons=[]
        if n<30:reasons.append("DEVELOPMENT_TRADES_BELOW_30")
        if not np.isfinite(mean) or mean<=0:reasons.append("DEVELOPMENT_MEAN_NET_NOT_POSITIVE")
        if score is None:reasons.append("DEVELOPMENT_RISK_ADJUSTED_SCORE_UNAVAILABLE")
        elif base_score is not None and score<=base_score:reasons.append("SCORE_NOT_ABOVE_63_SYMBOL_BASELINE")
        elif base_score is None:reasons.append("63_SYMBOL_BASELINE_SCORE_UNAVAILABLE")
        row["recommendation_check"]={"development_trades":n,
            "development_mean_net_pct":round(mean,8) if np.isfinite(mean) else None,
            "development_sequence_mdd_pct":round(mdd,8) if np.isfinite(mdd) else None,
            "development_risk_adjusted_score":round(score,10) if score is not None else None,
            "beats_63_symbol_development_score":bool(score is not None and base_score is not None and score>base_score),
            "qualifies":bool(qualifies),"exclusion_reasons":reasons}
        table.append({"set_name":name,"total_symbols_n":row["total_symbols_n"],
            **row["recommendation_check"]})
        if qualifies:candidates.append((row["total_symbols_n"],-score,name,row))
    table.sort(key=lambda r:(r["total_symbols_n"],r["set_name"]))
    baseline_info={"trades":base_n,"mean_net_pct":round(base_mean,8) if np.isfinite(base_mean) else None,
        "sequence_mdd_pct":round(base_mdd,8) if np.isfinite(base_mdd) else None,
        "risk_adjusted_score":round(base_score,10) if base_score is not None else None}
    if not candidates:
        return {"status":"KEEP_63_SYMBOL_BENCHMARK","selected_set":None,
            "conclusion":"개발기간 기준으로는 63종목 벤치마크 유지를 권고합니다.",
            "reason":"조건(개발 거래 30건 이상, 평균 순손익 양수, 개발 위험조정 점수가 63종목보다 큼)을 모두 만족한 축소 세트가 없습니다.",
            "recommendation_score_formula":"mean_net_pct / abs(sequence MDD)",
            "selection_used_validation_or_diagnostic":False,
            "baseline_63_development":baseline_info,"candidate_development_table":table}
    _,_,name,row=sorted(candidates,key=lambda c:(c[0],c[1],c[2]))[0]
    selected=row["selected_tickers"]
    conclusion=(f"개발기간 추천: 코스피 {len(selected['KOSPI'])} / 코스닥 {len(selected['KOSDAQ'])} "
        f"/ TIGER ETF {len(selected['TIGER_ETF'])} / SOXL {len(selected['SOXL'])} "
        f"(총 {row['total_symbols_n']}종목, {name}).")
    return {"status":"FOCUSED_UNIVERSE_RECOMMENDED","selected_set":name,
        "conclusion":conclusion,"reason":"개발기간 조건을 충족하고 63종목보다 높은 위험조정 점수를 낸 세트 중 실제 종목 수가 가장 작습니다.",
        "recommendation_score_formula":"mean_net_pct / abs(sequence MDD)",
        "selection_used_validation_or_diagnostic":False,"baseline_63_development":baseline_info,
        "recommended_development":row["recommendation_check"],
        "recommended_tickers":selected,"candidate_development_table":table}


def _focused_universe_ablation(cohorts,frames_by_cohort,policy_trades_by_cohort,policy_info):
    rankings={}
    for cohort in ("KOSPI","KOSDAQ"):
        rankings[cohort]=_focused_liquidity_ranking(cohort,cohorts.get(cohort,[]),
            frames_by_cohort.get(cohort,{}))
    liquid={cohort:list(row["eligible_tickers_by_liquidity"]) for cohort,row in rankings.items()}
    fixed={cohort:list(AUDIT_63_BENCHMARK[cohort]) for cohort in ("KOSPI","KOSDAQ","TIGER_ETF","SOXL")}
    specs={"baseline_63":{"selected":fixed,"kind":"frozen_63_baseline","recommendation_candidate":False}}
    for kp in FOCUSED_UNIVERSE_SIZES:
        for kq in FOCUSED_UNIVERSE_SIZES:
            name=f"liquidity_top_KOSPI{kp}_KOSDAQ{kq}"
            specs[name]={"selected":{"KOSPI":liquid["KOSPI"][:kp],"KOSDAQ":liquid["KOSDAQ"][:kq],
                "TIGER_ETF":fixed["TIGER_ETF"],"SOXL":fixed["SOXL"]},
                "kind":"liquidity_top_grid","requested_sizes":{"KOSPI":kp,"KOSDAQ":kq},
                "recommendation_candidate":True}
    specs["liquidity_top_only_10_each"]={"selected":{"KOSPI":liquid["KOSPI"][:10],
        "KOSDAQ":liquid["KOSDAQ"][:10],"TIGER_ETF":fixed["TIGER_ETF"],"SOXL":fixed["SOXL"]},
        "kind":"fixed_liquidity_top_only","requested_sizes":{"KOSPI":10,"KOSDAQ":10},
        "recommendation_candidate":True}
    # This named comparison is intentionally anchored on the three requested
    # core holdings. Add only eligible development-liquidity names around them.
    core=[t for t in FOCUSED_CORE_KOSPI if t in fixed["KOSPI"]]
    core_eligible=[t for t in core if t in liquid["KOSPI"]]
    core_set=core+ [t for t in liquid["KOSPI"] if t not in core][:max(0,10-len(core))]
    specs["core3_plus_liquidity_fill10"]={"selected":{"KOSPI":core_set,
        "KOSDAQ":liquid["KOSDAQ"][:10],"TIGER_ETF":fixed["TIGER_ETF"],"SOXL":fixed["SOXL"]},
        "kind":"fixed_core_plus_development_liquidity","requested_sizes":{"KOSPI":10,"KOSDAQ":10},
        "core_tickers_requested":list(FOCUSED_CORE_KOSPI),"core_tickers_eligible":core_eligible,
        "core_tickers_included":core,
        "recommendation_candidate":True}
    specs["kospi_only_kosdaq_excluded"]={"selected":{"KOSPI":liquid["KOSPI"][:30],
        "KOSDAQ":[],"TIGER_ETF":fixed["TIGER_ETF"],"SOXL":fixed["SOXL"]},
        "kind":"fixed_kosdaq_excluded","requested_sizes":{"KOSPI":30,"KOSDAQ":0},
        "recommendation_candidate":True}
    all_frames={}
    for frame_map in frames_by_cohort.values():all_frames.update(frame_map)
    name_by_cohort={cohort:{str(row[0]):str(row[1]) for row in items}
        for cohort,items in cohorts.items()}
    all_items={cohort:list(AUDIT_63_BENCHMARK[cohort]) for cohort in fixed}
    for cohort in fixed:
        all_items[cohort]=list(fixed[cohort])
    ranking_by_ticker={cohort:{r["ticker"]:r for r in ranking["symbols"]}
        for cohort,ranking in rankings.items()}
    baseline_periods={}
    for period in ("development","validation","diagnostic"):
        baseline_periods[period]=_resolve_nonoverlapping(
            [trade for cohort in fixed for trade in policy_trades_by_cohort.get(cohort,{}).get(period,[])])
    baseline_counts={p:len(rows) for p,rows in baseline_periods.items()}
    reports={}
    for name,spec in specs.items():
        selected={cohort:list(spec["selected"].get(cohort,[])) for cohort in fixed}
        selected_sets={cohort:set(codes) for cohort,codes in selected.items()}
        requested=sum(len(codes) for codes in selected.values())
        selected_all=set().union(*selected_sets.values())
        exclusions={};evidence={}
        for cohort in fixed:
            exclusions[cohort]=[];evidence[cohort]=[]
            selected_codes=selected_sets[cohort]
            if cohort in ranking_by_ticker:
                for ticker in selected[cohort]:
                    evidence_row=ranking_by_ticker[cohort].get(ticker,{
                        "ticker":ticker,"eligible":False,"exclusion_reasons":["NOT_IN_FROZEN_COHORT_METADATA"]})
                    if (spec["kind"]=="fixed_core_plus_development_liquidity" and cohort=="KOSPI"
                            and ticker in FOCUSED_CORE_KOSPI):
                        evidence_row={**evidence_row,"inclusion_override":"PREDECLARED_CORE_TICKER"}
                    evidence[cohort].append(evidence_row)
            for ticker in all_items[cohort]:
                if ticker in selected_codes:continue
                rankrow=ranking_by_ticker.get(cohort,{}).get(ticker)
                if rankrow and not rankrow["eligible"]:
                    reason="DATA_QUALITY_FILTER:"+";".join(rankrow["exclusion_reasons"])
                elif cohort=="KOSDAQ" and spec["kind"]=="fixed_kosdaq_excluded":
                    reason="KOSDAQ_EXCLUDED_BY_PREDECLARED_FIXED_VARIANT"
                elif cohort in ("KOSPI","KOSDAQ"):
                    reason="LOWER_DEVELOPMENT_LIQUIDITY_THAN_SET_CUTOFF"
                else:reason="OUTSIDE_SET_DEFINITION"
                exclusions[cohort].append({"ticker":ticker,"name":name_by_cohort.get(cohort,{}).get(ticker,ticker),
                    "reason":reason,"liquidity_rank":rankrow.get("liquidity_rank") if rankrow else None})
        period_metrics={}
        for period in ("development","validation","diagnostic"):
            base_rows=baseline_periods[period]
            selected_rows=_resolve_nonoverlapping([t for t in base_rows if t.get("ticker") in selected_all])
            metric=_metrics_with_portfolio(selected_rows,requested,all_frames,period)
            metric["policy_entry_acceptance_rate_pct"]=(round(100*len(selected_rows)/baseline_counts[period],4)
                if baseline_counts[period] else 0.0)
            metric["policy_entries_in_63_baseline_n"]=baseline_counts[period]
            metric["small_sample"]=period=="validation" and metric.get("trades",0)<MIN_POLICY_TRADES
            period_metrics[period]=metric
        item={"kind":spec["kind"],"requested_sizes":spec.get("requested_sizes"),
            "recommendation_candidate":spec.get("recommendation_candidate",False),
            "total_symbols_n":requested,"selected_tickers":selected,
            "selection_evidence":evidence,"excluded_tickers_by_cohort":exclusions,
            "periods":period_metrics}
        if "core_tickers_requested" in spec:
            item["core_tickers_requested"]=spec["core_tickers_requested"]
            item["core_tickers_eligible"]=spec["core_tickers_eligible"]
            item["core_tickers_included"]=spec["core_tickers_included"]
        reports[name]=item
    selection_info={cohort:{"selected_policy":policy_info.get(cohort,{}).get("selected_policy"),
        "selected_parameters":policy_info.get(cohort,{}).get("selected_parameters"),
        "source":"63-symbol cohort development-only selected legacy policy; parameters frozen for all subsets"}
        for cohort in fixed}
    recommendation=_focused_recommendation(reports,reports["baseline_63"]["periods"]["development"])
    return {"benchmark_unchanged":True,"sample_source":"already-loaded frozen 63 benchmark OHLCV only; no additional instrument fetches",
        "selection_rule":{"liquidity_rank":"descending median daily close×volume, development only; ticker ascending ties",
            "eligibility":{"development_window_start":FOCUSED_HISTORY_START,"development_end_exclusive":TRAIN_END,
                "data_completeness_pct_min":FOCUSED_MIN_DATA_COMPLETENESS*100,
                "first_development_bar_before":FOCUSED_FIRST_BAR_CUTOFF,
                "minimum_development_bars":FOCUSED_MIN_DEVELOPMENT_BARS},
            "universe_sizes_per_korean_cohort":list(FOCUSED_UNIVERSE_SIZES),
            "etf_soxl_always_retained":True,
            "policy":"63-universe selected A-D policy and parameters frozen by cohort; no fitting or selection on validation/diagnostic"},
        "selected_policy_by_cohort":selection_info,"development_liquidity_evidence":rankings,
        "sets":reports,"recommended_universe":recommendation}

def _model_diagnostics(rows):
    return {"r_rank_metrics":_rank_metrics(rows),"auroc":_auc(rows),
            "brier":_brier(rows,"model_probability"),"ece":_calibration(rows).get("ece"),
            "probability_spread":_probability_spread(rows),"calibration":_calibration(rows)}

def run():
    started=datetime.now(timezone.utc)
    universe=_audit_universe()
    cohorts=_frozen_audit_cohorts(universe)
    macro_frames,macro_sources=get_market_series_bundle(start_date="2016-01-01",end_date=MIN_END_DATE)
    baseline_path=Path("data/cec6d80-baseline.json")
    archived_baseline=json.loads(baseline_path.read_text(encoding="utf-8")) if baseline_path.exists() else {}
    out={"created_utc":started.isoformat(),"status":"RESEARCH_ONLY_NOT_LIVE_APPROVED",
        "sample_start":SAMPLE_START,"sample_start_by_cohort":{},
        "windows":{"development_end":TRAIN_END,"validation_end":VALID_END,"diagnostic_end":HOLDOUT_END},
        "rank_target": "realized net return percent / entry net downside percent",
        "feature_timing":"all model features and planned payoff/risk fields use signal-day close; next-open values are outcome/execution only",
        "model_comparison":{},"r_rank_metrics":{},"acceptance_rate":{},
        "regime_features":{"definitions":REGIME_RULE,"market_series_sources":macro_sources,
            "us_close_shift":"Korean signal dates use the most recent US index date strictly earlier than the Korean signal date; SOXL may use the same US close. This prevents using a later US close.",
            "macro_source_preference":"FinanceDataReader first; Yahoo Finance retries as fallback; source failures and partial coverage are recorded.",
            "fixed_scale":True,"evaluation_period_scaling":False},
        "regime_distribution":{},"regime_performance":{},"ablation":{},
        "feature_coverage":{},"target_variants":{},"policy_comparison":{},"policy_ladder":{},
        "disclosure_coverage":{},"focused_universe_ablation":{},"recommended_universe":{},
        "direction3_role":"diagnostic_only; all buy decisions come exclusively from the legacy policy ladder",
        "frozen_audit_benchmark":{k:list(v) for k,v in AUDIT_63_BENCHMARK.items()},
        "feature_schema":{"version":"macro-relative-flow-v1","groups":{k:list(v) for k,v in FEATURE_GROUPS.items()},
            "full_numeric_feature_count":len(NUMERIC_FEATURES),"strategy_flags":6,"strategy_interactions":6,
            "feature_count_including_strategy_terms":len(NUMERIC_FEATURES)+12,"feature_limit":90,
            "fixed_centers":CENTERS,"fixed_scales":SCALES,
            "limit_rationale":"59 fixed-scale numeric values + six strategy one-hots + six strategy×net-RR interactions = 71 fixed columns; schema fixed before OOS and no evaluation-window fitting/selection."},
        "cec6d80_baseline_source":archived_baseline.get("source_run_id"),
        "portfolio":{"max_concurrent_positions":5,"weighting":"equal initial notional slots"},"cohorts":{}}
    all_flow_frames={};all_flow_sources={};all_development_feature_rows=[]
    focused_frames_by_cohort={};focused_policy_trades_by_cohort={};focused_policy_info={}
    disclosure_entries=[];seen_disclosure_tickers=set()
    for cohort in ("KOSPI","KOSDAQ"):
        for ticker,name,_,_ in cohorts[cohort]:
            if ticker not in seen_disclosure_tickers:
                disclosure_entries.append((ticker,name));seen_disclosure_tickers.add(ticker)
    disclosure_by_ticker={}
    for ticker,name in disclosure_entries:
        result=get_cached_company_disclosures(ticker,name,start_date="2015-12-01",end_date=MIN_END_DATE)
        disclosure_by_ticker[ticker]=result
        diag=result.get("diagnostics",{})
        print(f"DISCLOSURE_CACHE ticker={ticker} rows={len(result.get('records',[]))} "
            f"available={diag.get('available',False)} partial={diag.get('partial',False)} "
            f"year_coverage={diag.get('year_coverage_pct')}",flush=True)
    available_disclosure_symbols=sum(bool(v.get("diagnostics",{}).get("available"))
        for v in disclosure_by_ticker.values())
    partial_disclosure_symbols=[{"ticker":ticker,
        "year_coverage_pct":v.get("diagnostics",{}).get("year_coverage_pct"),
        "available":bool(v.get("diagnostics",{}).get("available")),
        "failures":v.get("diagnostics",{}).get("failures",[])}
        for ticker,v in disclosure_by_ticker.items() if v.get("diagnostics",{}).get("partial",False)]
    disclosure_collection={"requested_symbols":len(disclosure_entries),"attempted_symbols":0,
        "available_symbols":available_disclosure_symbols,"partial_symbols":partial_disclosure_symbols,
        "request_interval_seconds":request_interval_seconds(),"budget_seconds":0,
        "budget_used_seconds":0.0,"budget_exhausted":False,"workers":0,
        "cache_only":True,"network_requests_in_audit":0,
        "collection_guidance":"Run disclosure-warmup manually in smaller batches over several days; cached completed years are reused and failed/missing year-pages resume later.",
        "failures":[{"ticker":ticker,"error":"unavailable cached disclosure history",
            "failures":v.get("diagnostics",{}).get("failures",[])}
            for ticker,v in disclosure_by_ticker.items() if not v.get("diagnostics",{}).get("available",False)]}
    out["disclosure_collection"]={**disclosure_collection,"source":"DART public HTML detailSearch.ax",
        "requested_start":"2015-12-01","requested_end":MIN_END_DATE,
        "throttling":"DART HTTP 503 under bulk collection; cache resumes failed year-pages on later runs",
        "partial_symbols_n":len(disclosure_collection.get("partial_symbols",[])),
        "failure_reasons_top5":_disclosure_failure_reasons_top(disclosure_by_ticker,limit=5),
        "by_ticker":{code:{"first_date":v.get("diagnostics",{}).get("first_date"),
            "last_date":v.get("diagnostics",{}).get("last_date"),"rows_n":len(v.get("records",[])),
            "requests_n":v.get("diagnostics",{}).get("requests_n",0),
            "year_coverage_pct":v.get("diagnostics",{}).get("year_coverage_pct"),
            "available_years_n":v.get("diagnostics",{}).get("available_years_n"),
            "requested_years_n":v.get("diagnostics",{}).get("requested_years_n"),
            "available":v.get("diagnostics",{}).get("available",False),
            "partial":v.get("diagnostics",{}).get("partial",False),
            "complete":v.get("diagnostics",{}).get("complete",False),
            "failures":v.get("diagnostics",{}).get("failures",[])}
            for code,v in disclosure_by_ticker.items()},
        "unavailable_tickers":sorted(code for code,v in disclosure_by_ticker.items()
            if not v.get("diagnostics",{}).get("available",False)),
        "partial_tickers":sorted(code for code,v in disclosure_by_ticker.items()
            if v.get("diagnostics",{}).get("partial",False)),
        "not_applicable_tickers":{code:"ETF/US security outside the Korean ordinary-stock DART feature universe"
            for cohort in ("TIGER_ETF","SOXL") for code,_,_,_ in cohorts[cohort]}}
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
        flow_frames={};flow_sources={}
        if cohort=="SOXL":
            flow_sources={"SOXL":{"status":"not_applicable","source":None,
                "ticker":"SOXL","attempted":False,"available":False,"requests_n":0,"rows_n":0,
                "first_date":None,"last_date":None,
                "reason":"US-listed security; Korean foreign/institution flow source does not apply. flow_data_available is explicitly zero."}}
        else:
            with ThreadPoolExecutor(max_workers=12) as executor:
                future_to_ticker={}
                for ticker,_,_,_ in items:
                    if ticker not in frames:
                        flow_sources[ticker]={"status":"unavailable","source":None,"attempted":False,"available":False,
                            "requests_n":0,"rows_n":0,"first_date":None,"last_date":None,
                            "reason":"OHLCV history unavailable"}
                    elif pd.Timestamp(frames[ticker].index[0]).date()>pd.Timestamp(HOLDOUT_END).date():
                        flow_sources[ticker]={"status":"unavailable","source":None,"attempted":False,"available":False,
                            "requests_n":0,"rows_n":0,"first_date":None,"last_date":None,
                            "reason":"listing begins after the fixed holdout window"}
                    else:
                        future_to_ticker[executor.submit(get_historical_investor_flows,ticker,
                            start_date="2015-12-01",end_date=MIN_END_DATE)]=ticker
                for future in as_completed(future_to_ticker):
                    ticker=future_to_ticker[future]
                    try:
                        flow_frames[ticker],source=future.result()
                        diag=flow_frames[ticker].attrs.get("collection_diagnostics",{})
                        flow_sources[ticker]={**diag,"status":"ok","source":source,
                            "observations":len(flow_frames[ticker]),"available":True}
                    except Exception as exc:
                        diag=getattr(exc,"diagnostics",{})
                        flow_sources[ticker]={**diag,"ticker":ticker,"status":"unavailable","source":None,
                            "attempted":diag.get("attempted",True),"available":False,"reason":str(exc)[:500]}
        all_flow_frames.update(flow_frames)
        all_flow_sources.update(flow_sources)
        context=build_market_context(frames,macro_frames,cohort,flow_frames)
        for inst in instruments:inst["investor_flow_source"]=flow_sources.get(inst["ticker"],{})
        breadth,gap,market_return60=_make_cohort_context(frames)
        def attach_disclosure(row):
            record=disclosure_by_ticker.get(row["ticker"],{})
            diag=record.get("diagnostics",{});available=bool(diag.get("available",False) and record.get("records"))
            features=disclosure_features_for_signal(record.get("records",[]),row["signal_date"],
                frames[row["ticker"]].index,available=available)
            row.update(features);row["disclosure_data_available"]=available
            row["disclosure_data_partial"]=bool(diag.get("partial",False))
            return row
        legacy_all=[]
        for strategy,ts in legacy_by_variant.items():
            for t in ts:
                frame=frames.get(t["ticker"]);comp=computed_by.get(t["ticker"])
                if frame is not None:
                    row=_decorate_features({**t,"candidate_family":"legacy","candidate_variant":strategy,"strategy":strategy},frame,comp,breadth,gap,market_return60)
                    row.update(context["features"].get(t["ticker"],{}).get(t["signal_date"],{}))
                    attach_disclosure(row)
                    legacy_all.append(row)
        broad_all=[]
        for t in broad:
            frame=frames.get(t["ticker"]);comp=computed_by.get(t["ticker"])
            if frame is not None:
                row=_decorate_features({**t,"candidate_family":"broad","candidate_variant":"broad_market_ml","strategy":"broad_market_ml"},frame,comp,breadth,gap,market_return60)
                row.update(context["features"].get(t["ticker"],{}).get(t["signal_date"],{}))
                attach_disclosure(row)
                broad_all.append(row)
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
        out["sample_start_by_cohort"][cohort]["cross_sectional_rank_counts"]=context["cross_sectional_rank_counts"]
        out["sample_start_by_cohort"][cohort]["flow_source_coverage"]={
            "available_symbols":sum(v.get("available",v.get("status")=="ok") for v in flow_sources.values()),
            "attempted_symbols":sum(bool(v.get("attempted",False)) for v in flow_sources.values()),
            "sources":flow_sources}
        out["regime_distribution"][cohort]={}
        dev_variant={s:_trade_metrics(_period(ts,"development")) for s,ts in legacy_by_variant.items()}
        valid_variants=[s for s,m in dev_variant.items() if m["trades"]>=MIN_POLICY_TRADES and m.get("mean_net_pct") is not None]
        legacy_choice=max(valid_variants,key=lambda s:(dev_variant[s]["mean_net_pct"],s),default=None)
        legacy_choice_rows=[r for r in legacy_all if r.get("strategy")==legacy_choice] if legacy_choice else []
        missing_disclosure_tickers=sorted(ticker for ticker,_,_,_ in items
            if not disclosure_by_ticker.get(ticker,{}).get("diagnostics",{}).get("available",False))
        partial_disclosure_tickers=sorted(ticker for ticker,_,_,_ in items
            if disclosure_by_ticker.get(ticker,{}).get("diagnostics",{}).get("partial",False))
        available_disclosure_n=len(items)-len(missing_disclosure_tickers)
        cohort_disclosure_coverage={"available_symbols_n":available_disclosure_n,
            "requested_symbols_n":len(items),
            "coverage_pct":100.0*available_disclosure_n/len(items) if items else 0.0,
            "partial_symbols_n":len(partial_disclosure_tickers)}
        ladder=_policy_ladder(legacy_choice_rows,frames,len(items),cohort,cohort_disclosure_coverage)
        out["policy_ladder"][cohort]=ladder
        disclosure_fields=("disclosure_periodic_days_since","disclosure_bad_20d","disclosure_bad_60d",
            "disclosure_positive_20d","disclosure_count_60d")
        disclosure_coverage={}
        for period_name in ("development","validation","diagnostic"):
            period_candidates=_period(legacy_all+broad_all,period_name)
            disclosure_coverage[period_name]={}
            for feature in disclosure_fields:
                observed=[]
                for row in period_candidates:
                    try:value=float(row.get(feature,np.nan))
                    except (TypeError,ValueError):value=np.nan
                    if np.isfinite(value):observed.append(value)
                disclosure_coverage[period_name][feature]={"rows_n":len(period_candidates),
                    "non_missing_n":len(observed),"non_missing_pct":round(100*len(observed)/len(period_candidates),2) if period_candidates else 0.0}
        disclosure_coverage_summary={
            "periods":disclosure_coverage,"data_missing_tickers":missing_disclosure_tickers,
            "partial_tickers":partial_disclosure_tickers,
            "available_symbols_n":available_disclosure_n,"requested_symbols_n":len(items),
            "available_symbol_coverage_pct":round(cohort_disclosure_coverage["coverage_pct"],2),
            "partial_symbols_n":len(partial_disclosure_tickers),
            "abstain_when_unavailable":True,
            "bad_title_keywords":list(BAD_KEYWORDS),"bad_title_exclusions":list(BAD_TITLE_EXCLUSIONS),
            "disclosure_bad_title_counts":_disclosure_bad_title_counts(items,disclosure_by_ticker,limit=10)}
        out.setdefault("disclosure_coverage",{})[cohort]=disclosure_coverage_summary
        out.setdefault("disclosure_feature_coverage",{})[cohort]=disclosure_coverage_summary
        development_rows=_period(eligible_rows,"development")
        all_development_feature_rows.extend(development_rows)
        feature_coverage,excluded_features=_feature_coverage(development_rows)
        for col in excluded_features:print(f"FEATURE_AUTO_EXCLUDED cohort={cohort} feature={col} reason=zero_development_std",flush=True)
        excluded_numeric_count=len(set(excluded_features)&set(NUMERIC_FEATURES))
        excluded_strategy_count=len(excluded_features)-excluded_numeric_count
        active_strategy_terms=12-excluded_strategy_count
        target_results={};target_status={}
        min_train=MIN_TRAIN_SAMPLES_BY_COHORT[cohort]
        for target_variant in ("pooled","date_cs"):
            scored_configs={};statuses={};dev_comp={}
            for config in MODEL_CONFIGS:
                scored_configs[config],statuses[config]=walk_forward_scores(
                    eligible_rows,config,min_train,target_variant=target_variant,
                    excluded_features=excluded_features)
                dev=_period(scored_configs[config],"development")
                rank=_rank_metrics(dev)
                dev_comp[config]={"r_rank_metrics":rank,"datewise_primary_metric":rank.get("datewise_spearman_mean"),
                    "auroc":_auc(dev),"candidate_n":len(dev),"minimum_train_samples":min_train,
                    "max_training_samples":4000,"numeric_features":len(NUMERIC_FEATURES)-excluded_numeric_count,
                    "auto_excluded_features":excluded_features,
                    "feature_count_including_strategy_terms":len(NUMERIC_FEATURES)-excluded_numeric_count+active_strategy_terms}
            selectable=[(v["datewise_primary_metric"],v["r_rank_metrics"].get("tercile_top_minus_bottom_realized_r"),k)
                for k,v in dev_comp.items() if v["datewise_primary_metric"] is not None and v["candidate_n"]>=MIN_POLICY_TRADES]
            chosen=max(selectable,key=lambda p:(p[0],p[1] if p[1] is not None else -1,p[2]))[2] if selectable else None
            target_results[target_variant]={"selected_config":chosen,"model_comparison":dev_comp,
                "score_status":statuses,"selected_scores":scored_configs.get(chosen,[]),
                "selected_dev_metrics":dev_comp.get(chosen,{}).get("r_rank_metrics")}
        target_choices=_target_variant_choices(target_results)
        selected_target=max(target_choices,key=lambda x:(x[0],x[1]))[1] if target_choices else "date_cs"
        selected_config=target_results[selected_target]["selected_config"]
        scored=target_results[selected_target]["selected_scores"]
        policy_scored=target_results["date_cs"]["selected_scores"]
        policy_thresholds=_datewise_thresholds(_period(policy_scored,"development")) if target_results["date_cs"]["selected_config"] else {}
        dev_scored=_period(scored,"development")
        thresholds=_datewise_thresholds(dev_scored) if selected_config else {}
        selected_dev_metrics=target_results[selected_target].get("selected_dev_metrics") or {}
        selected_dev_date_ic=selected_dev_metrics.get("datewise_spearman_mean")
        model_policy_table={};model_policy=None
        date_cs_status="NO_DEVELOPMENT_EDGE" if selected_dev_date_ic is None or selected_dev_date_ic<=0 else "DEVELOPMENT_EDGE_CANDIDATE"
        for name,spec in thresholds.items():
            if not name.startswith("r_p"):continue
            passed=_datewise_selected(dev_scored,name,thresholds)
            trades=_resolve_nonoverlapping(passed)
            model_policy_table[name]={"threshold_predicted_r":spec.get("global"),
                "accepted_candidate_n":len(passed),"acceptance_rate_pct":round(100*len(passed)/len(dev_scored),2) if dev_scored else 0,
                "trades":_trade_metrics(trades),"rank_metrics":_rank_metrics(passed)}
        eligible_model_policies=[(v["trades"].get("mean_net_pct"),name) for name,v in model_policy_table.items()
            if selected_dev_date_ic is not None and selected_dev_date_ic>0
            and v["trades"].get("trades",0)>=MIN_POLICY_TRADES
            and v["trades"].get("mean_net_pct") is not None and v["trades"]["mean_net_pct"]>MIN_DEVELOPMENT_EDGE]
        if eligible_model_policies:model_policy=max(eligible_model_policies,key=lambda x:(x[0],x[1]))[1]
        model_policy_status="DEVELOPMENT_EDGE_SELECTED" if model_policy else (
            "NO_DEVELOPMENT_EDGE" if date_cs_status=="NO_DEVELOPMENT_EDGE" else "INSUFFICIENT_DEVELOPMENT_SAMPLE")
        model_candidates_before_gate=[]
        for period_name in ("development","validation","diagnostic"):
            period_rows=_period(scored,period_name)
            if model_policy:
                model_candidates_before_gate.extend(_datewise_selected(period_rows,model_policy,thresholds))
        model_trades_before_gate=_resolve_nonoverlapping(model_candidates_before_gate)
        model_candidates_after_gate=[t for t in model_candidates_before_gate if not _broad_risk_off_blocked(t)]
        model_trades=_resolve_nonoverlapping(model_candidates_after_gate)
        dev_legacy=_resolve_nonoverlapping(_period([r for r in legacy_all if r.get("strategy")==legacy_choice],"development")) if legacy_choice else []
        policy_comparison={"periods":{},"development_gate":{},"selected_policy":None}
        for period_name in ("development","validation","diagnostic"):
            legacy_period=_period([r for r in legacy_all if r.get("strategy")==legacy_choice],period_name) if legacy_choice else []
            scored_period=_period(policy_scored,period_name)
            policy_comparison["periods"][period_name]=_policy_candidate_metrics(
                legacy_period,scored_period,frames,len(items),period_name,policy_thresholds)
        base_dev=policy_comparison["periods"]["development"]["legacy_only"]["trades"].get("mean_net_pct")
        for name,entry in policy_comparison["periods"]["development"].items():
            if name=="legacy_only":entry["development_gate_passed"]=True;continue
            mean_net=entry["trades"].get("mean_net_pct")
            rank_ic=(entry.get("datewise_rank_metrics") or {}).get("datewise_spearman_mean")
            improves=mean_net is not None and base_dev is not None and mean_net>base_dev
            sufficient=entry["trades"].get("trades",0)>=MIN_POLICY_TRADES
            entry["development_gate_passed"]=bool(sufficient and (improves or (rank_ic is not None and rank_ic>0)))
            entry["development_gate_reason"]="development net improves legacy" if improves else (
                "development datewise IC > 0" if rank_ic is not None and rank_ic>0 else "no development edge")
        eligible_legacy_policies=[(e["trades"].get("mean_net_pct"),name) for name,e in policy_comparison["periods"]["development"].items()
            if e.get("development_gate_passed") and e["trades"].get("mean_net_pct") is not None]
        score_filter_diagnostic_choice=max(eligible_legacy_policies,key=lambda x:(x[0],x[1]))[1] if eligible_legacy_policies else None
        policy_comparison["diagnostic_only_score_filter_choice"]=score_filter_diagnostic_choice
        policy_comparison["selected_policy"]=None
        policy_comparison["development_gate"]={name:e.get("development_gate_passed") for name,e in policy_comparison["periods"]["development"].items()}
        out.setdefault("policy_comparison",{})[cohort]=policy_comparison
        selected_legacy_policy=ladder["selected_policy"]
        selected_stage_for_focus=ladder.get(selected_legacy_policy,{})
        selected_params_for_focus=selected_stage_for_focus.get("params") or {}
        focused_frames_by_cohort[cohort]=frames
        focused_policy_info[cohort]={"selected_policy":selected_legacy_policy,
            "selected_parameters":selected_params_for_focus}
        focused_policy_trades_by_cohort[cohort]={period:_legacy_policy_stage(
            _period(legacy_choice_rows,period),selected_legacy_policy,selected_params_for_focus)
            for period in ("development","validation","diagnostic")}
        out.setdefault("feature_coverage",{})[cohort]={"development_rows_n":len(development_rows),
            "by_feature":feature_coverage,"auto_excluded_constant_features":excluded_features,
            "exclusion_basis":"development candidates only; evaluation-period variation never affects feature inclusion"}
        target_report={}
        for target_name,target_data in target_results.items():
            target_rows=target_data["selected_scores"]
            target_report[target_name]={"selected_config":target_data["selected_config"],
                "model_comparison":target_data["model_comparison"],
                "selected_dev_metrics":target_data["selected_dev_metrics"],
                "selected_period_metrics":{p:{"r_rank_metrics":_rank_metrics(_period(target_rows,p)),
                    "auroc":_auc(_period(target_rows,p)),"brier":_brier(_period(target_rows,p),"model_probability"),
                    "ece":_calibration(_period(target_rows,p)).get("ece"),
                    "predicted_r_spread":_model_diagnostics(_period(target_rows,p))["r_rank_metrics"].get("predicted_r_p10_p50_p90")}
                    for p in ("development","validation","diagnostic")}}
        out.setdefault("target_variants",{})[cohort]=target_report
        ablation_result={}
        if selected_config:
            for feature_set in FEATURE_GROUPS:
                ab_scored,_=walk_forward_scores(eligible_rows,selected_config,min_train,feature_set=feature_set,
                    target_variant=selected_target,excluded_features=excluded_features)
                active_numeric=sum(k not in excluded_features for k in FEATURE_GROUPS[feature_set])
                ablation_result[feature_set]={"model_config":selected_config,"target_variant":selected_target,
                    "feature_count":active_numeric+active_strategy_terms,"periods":{}}
                for period in ("development","validation","diagnostic"):
                    period_rows=_period(ab_scored,period)
                    ablation_result[feature_set]["periods"][period]={
                        "r_rank_metrics":_rank_metrics(period_rows),"auroc":_auc(period_rows),
                        "diagnostics":_model_diagnostics(period_rows)}
        out["ablation"][cohort]=ablation_result
        cohort_result={"requested":len(items),"loaded":len(instruments),"errors":errors,
            "data_coverage_pct":round(100*len(instruments)/len(items),2) if items else None,
            "data_complete":len(instruments)==len(items) and not errors,
            "model_eligible_symbols_n":len(eligible),"model_coverage_of_requested_pct":round(100*len(eligible)/len(items),2) if items else None,
            "candidate_pool":{"legacy":len(legacy_all),"broad":len(broad_all),"eligible_combined":len(eligible_rows)},
            "legacy_selected_development_strategy":legacy_choice,"legacy_development_by_variant":dev_variant,
            "model_comparison":target_results[selected_target]["model_comparison"],
            "target_variants":target_report,
            "selected_target_variant":selected_target,"selected_model":selected_config,
            "selected_hyperparameters":MODEL_CONFIGS.get(selected_config),
            "minimum_training_samples":MIN_TRAIN_SAMPLES_BY_COHORT[cohort],
            "model_selection_metric":"development mean of date-wise Spearman only; dates with 1-2 candidates excluded; pooled R-IC is secondary",
            "feature_coverage":feature_coverage,"auto_excluded_constant_features":excluded_features,
            "policy_selection_rank_ic_gate":{"criterion":"selected development date-wise Spearman > 0",
                "development_datewise_spearman":selected_dev_date_ic,
                "passed":bool(selected_dev_date_ic is not None and selected_dev_date_ic>0)},
            "selected_model_policy":model_policy,"selected_model_policy_thresholds":thresholds,
            "selected_model_status":model_policy_status,"development_predicted_r_thresholds":thresholds,
            "development_policy_comparison":model_policy_table,
            "selected_legacy_policy":selected_legacy_policy,"legacy_policy_comparison":policy_comparison,
            "market_context":{"regime_day_count":len(context["regime_by_date"]),
                "cross_sectional_rank_counts":context["cross_sectional_rank_counts"],
                "investor_flow_coverage":out["sample_start_by_cohort"][cohort]["flow_source_coverage"]},
            "periods":{},"instruments":[]}
        for period in ("development","validation","diagnostic"):
            # Use the decorated candidate rows here so the signal-time regime
            # tag survives into the legacy-vs-model regime performance split.
            legacy_pool=_period([r for r in legacy_all if r.get("strategy")==legacy_choice],period) if legacy_choice else []
            legacy_chosen=_legacy_policy_rows(legacy_pool,_period(scored,period),"legacy_only")
            selected_stage=ladder.get(selected_legacy_policy,{})
            legacy_policy_trades=_legacy_policy_stage(legacy_pool,selected_legacy_policy,selected_stage.get("params") or {})
            model_pool=_period(model_trades,period)
            all_scored=_period(scored,period)
            accepted_before_gate=_datewise_selected(all_scored,model_policy,thresholds) if model_policy else []
            accepted_candidates=[r for r in accepted_before_gate if not _broad_risk_off_blocked(r)]
            legacy_m=_metrics_with_portfolio(legacy_chosen,len(items),frames,period)
            legacy_policy_m=_metrics_with_portfolio(legacy_policy_trades,len(items),frames,period)
            model_m=_metrics_with_portfolio(model_pool,len(items),frames,period)
            model_m.update(_coverage(model_pool,len(items)))
            model_m["acceptance_rate_pct"]=round(100*len(accepted_candidates)/len(all_scored),2) if all_scored else 0.0
            diag=_model_diagnostics(all_scored)
            cec6=None
            cohort_result["periods"][period]={"legacy":legacy_m,"selected_legacy_policy":legacy_policy_m,
                "selected_legacy_policy_name":selected_legacy_policy,"new_model":model_m,
                "policy_ladder_selected_metrics":selected_stage.get("periods",{}).get(period),
                "cec6d80_model":archived_baseline.get("cohorts",{}).get(cohort,{}).get("periods",{}).get(period,{}).get("model"),
                "cec6d80_legacy":archived_baseline.get("cohorts",{}).get(cohort,{}).get("periods",{}).get(period,{}).get("legacy"),
                "accepted_candidates_n":len(accepted_candidates),"scored_candidate_n":len(all_scored),
                "diagnostics":diag,"r_rank_metrics":_rank_metrics(all_scored),
                "regime_gate":{"accepted_before_gate_n":len(accepted_before_gate),
                    "accepted_after_gate_n":len(accepted_candidates),
                    "blocked_broad_risk_off_n":sum(_broad_risk_off_blocked(r) for r in accepted_before_gate),
                    "hard_gate":"RISK_OFF blocks every new-entry candidate, including legacy; only the legacy policy ladder can make buy decisions"},
                "unselected_diagnostic":{"legacy_candidate_n":len(_period(legacy_all,period)),"broad_candidate_n":len(_period(broad_all,period)),
                    "legacy_realized":_trade_metrics(_period(legacy_all,period)),"broad_realized":_trade_metrics(_period(broad_all,period))}}
            regime_counts=_period_regime_counts(eligible_rows,context,period)
            out["regime_distribution"][cohort][period]=regime_counts
            legacy_gate_trades=[t for t in legacy_chosen if not _broad_risk_off_blocked(t)]
            out["regime_performance"].setdefault(cohort,{})[period]={
                regime:{"distribution":regime_counts[regime],
                    "legacy":_trade_metrics([t for t in legacy_chosen if t.get("market_regime")==regime]),
                    "legacy_with_risk_off_hard_gate":_trade_metrics([t for t in legacy_gate_trades if t.get("market_regime")==regime]),
                    "model_before_hard_gate":_trade_metrics([t for t in model_trades_before_gate if _period([t],period) and t.get("market_regime")==regime]),
                    "model_after_hard_gate":_trade_metrics([t for t in model_trades if _period([t],period) and t.get("market_regime")==regime]),
                    "accepted_candidates_before_gate_n":sum(r.get("market_regime")==regime for r in accepted_before_gate),
                    "accepted_candidates_after_gate_n":sum(r.get("market_regime")==regime for r in accepted_candidates)}
                for regime in ("RISK_ON","NEUTRAL","RISK_OFF","UNKNOWN")}
        for inst in instruments:
            ticker=inst["ticker"]
            inst["legacy"]={period:_trade_metrics([t for t in _period(legacy_by_variant.get(legacy_choice,[]) if legacy_choice else [],period) if t["ticker"]==ticker]) for period in ("validation","diagnostic")}
            inst["model_eligible"] = ticker in eligible
            inst["development_candidate_n"]=symbol_dev.get(ticker,0)
            inst["model"]={period:_trade_metrics([t for t in model_trades if t["ticker"]==ticker and _period([t],period)]) for period in ("validation","diagnostic")}
            cohort_result["instruments"].append(inst)
        out["model_comparison"][cohort]=target_results[selected_target]["model_comparison"]
        out["r_rank_metrics"][cohort]={p:cohort_result["periods"][p]["r_rank_metrics"] for p in cohort_result["periods"]}
        out["acceptance_rate"][cohort]={p:cohort_result["periods"][p]["new_model"]["acceptance_rate_pct"] for p in cohort_result["periods"]}
        out["cohorts"][cohort]=cohort_result
        integrity &= cohort_result["data_complete"]
        print("AUDIT_RESULT "+json.dumps({"cohort":cohort,"loaded":len(instruments),"legacy":legacy_choice,
            "model":selected_config,"target":selected_target,"status":model_policy_status,
            "legacy_policy":selected_legacy_policy,"validation":cohort_result["periods"]["validation"],
            "diagnostic":cohort_result["periods"]["diagnostic"]},ensure_ascii=False),flush=True)
    focused= _focused_universe_ablation(cohorts,focused_frames_by_cohort,
        focused_policy_trades_by_cohort,focused_policy_info)
    out["focused_universe_ablation"]={k:v for k,v in focused.items() if k!="recommended_universe"}
    out["recommended_universe"]=focused["recommended_universe"]
    overall_flow_coverage,overall_flow_excluded=_feature_coverage(all_development_feature_rows)
    flow_feature_keys=[key for key in overall_flow_coverage if key in {
        "foreign_flow_5d_to_turnover","foreign_flow_20d_to_turnover",
        "institution_flow_5d_to_turnover","institution_flow_20d_to_turnover",
        "foreign_buy_streak","institution_buy_streak","flow_direction_agreement","flow_data_available"}]
    out["feature_coverage"]["all_cohorts"]={"development_rows_n":len(all_development_feature_rows),
        "by_feature":{key:overall_flow_coverage[key] for key in flow_feature_keys},
        "auto_excluded_constant_features":[key for key in overall_flow_excluded if key in flow_feature_keys],
        "scope":"combined audited Korean KOSPI/KOSDAQ, TIGER ETF, and SOXL development candidates; no missing-flow value is treated as observed zero"}
    out["data_integrity_status"]="PASS" if integrity else "FAIL"
    out["runtime_seconds"]=(datetime.now(timezone.utc)-started).total_seconds()
    Path("data").mkdir(exist_ok=True);Path("data/audit-63-latest.json").write_text(json.dumps(out,ensure_ascii=False,indent=2,default=str),encoding="utf-8")
    if not integrity:raise SystemExit("FAILED: incomplete 63-instrument OHLCV coverage; see data/audit-63-latest.json")
    print("AUDIT_EXPANDED_UNIVERSE_DIRECTION_START",flush=True)
    from expanded_direction_audit import run_expanded_direction_audit
    out["flow_collection"]={"source":"Naver mobile stock trend API; no KRX login/pykrx or paid source",
        "requested_start":"2015-12-01","priority":"frozen KOSPI30 + KOSDAQ30 + 2 TIGER ETFs",
        "available_symbols":sum(v.get("available",False) for v in all_flow_sources.values()),
        "attempted_symbols":sum(bool(v.get("attempted",False)) for v in all_flow_sources.values()),
        "by_ticker":all_flow_sources}
    expanded=run_expanded_direction_audit(flow_frames=all_flow_frames,flow_source_coverage=all_flow_sources,
        macro_frames=macro_frames,macro_diagnostics=macro_sources)
    out["universe_expansion"]=expanded["universe_expansion"]
    out["extended_pool_collection"]=expanded["extended_pool_collection"]
    out["flow_collection"]["extended_296_pool_attempted"]=bool(
        expanded["extended_pool_collection"].get("attempted_symbols",0))
    out["flow_collection"]["extended_pool_reason"]=(
        "Extended KOSPI/KOSDAQ ordinary-share flow collection ran inside direction3; unavailable or budget-deferred symbols remain NaN."
        if out["flow_collection"]["extended_296_pool_attempted"] else
        "No new extended-pool flow request was issued; provided benchmark flow frames were reused and missing symbols remain NaN.")
    out["flow_collection"]["extended_pool_collection"]=expanded["extended_pool_collection"]
    out["label_variants"]=expanded["label_variants"]
    out["direction3"]=expanded["direction3"]
    out["feature_coverage"]["expanded_direction_model"]=expanded["feature_coverage"]
    out["expanded_direction_selection_rule"]=expanded["selection_rule"]
    out["runtime_seconds"]=(datetime.now(timezone.utc)-started).total_seconds()
    Path("data/audit-63-latest.json").write_text(json.dumps(out,ensure_ascii=False,indent=2,default=str),encoding="utf-8")
    print(f"AUDIT_EXPANDED_UNIVERSE_COMPLETE requested={expanded['universe_expansion']['requested']} loaded={expanded['universe_expansion']['loaded']} failures={len(expanded['universe_expansion']['failures'])}",flush=True)
    return out

if __name__=="__main__":run()
