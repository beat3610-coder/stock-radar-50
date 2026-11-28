"""Universe expansion and causal three-way direction research for audit-63.

Research-only. Signals are observed at the close; entries use the next open.
Labels are revealed only after their horizon close and models refit annually
using labels available strictly before the signal date.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import date, timedelta
import json
import math
import os
from pathlib import Path
import re
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from scipy.optimize import minimize_scalar
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
    confusion_matrix, f1_score, log_loss, precision_recall_fscore_support,
    roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from providers import (get_historical_investor_flows, get_historical_prices,
                      get_market_series_bundle)
from universe import get_universe, MANDATORY_KR, MANDATORY_US, AUDIT_63_BENCHMARK
from engine import compute
from market_context import build_market_context

TRAIN_END="2025-11-01"; VALID_END="2026-04-01"; HOLDOUT_END="2026-11-01"
MIN_END_DATE=(date.fromisoformat(HOLDOUT_END)+timedelta(days=12)).isoformat()
SAMPLE_START="2016-01-01"; COST_PCT=.35; THETAS=(.3,.5,1.0)
HORIZONS=(5,10,20); MIN_TRAIN=500; REFIT_MONTH=24; SIGNAL_STEP=5
# band_regression maps a predicted volatility-scaled return through the same band
# that defines the label, so it uses magnitudes rather than only the class label.
# band_winsor clips the regression target at development tails and band_z regresses
# the volatility-normalised return, so neither is dominated by extreme moves.
MODEL_KINDS=("multinomial_logistic","random_forest","band_regression","band_winsor","band_z")
CAUSAL_PRIOR_WINDOW_DAYS=365
# Shrinkage target for probability calibration: class frequencies of labels that
# were already observable before each signal date (chosen on development only).
PRIOR_SHRINKAGE_WINDOW_DAYS=250
MIN_PRIOR_ROWS=30
DECISION_TAU_GRID=(0.,.05,.1,.15,.2,.25,.3,.35,.4,.45,.5,.55,.6,.65,.7,.75,.8,.85,.9,.95,1.,1.05,1.1,1.15,1.2)
SIDEWAYS_TARGET_SHARE=.30
MIN_DEVELOPMENT_IC=.01
# Direction label mode per cohort. "absolute_return" keeps the existing label:
# the stock's own net return after the existing cost against the frozen band.
# "excess_vs_cohort_mean" labels the stock's net return minus the equal-weighted
# net return of the same cohort on the same signal date and horizon, i.e. whether
# it beat its own market rather than whether the market rose. The basket mean is
# built from the same exit closes the member labels need, so it becomes knowable
# exactly on each row's own label_available_date and never earlier. KOSPI keeps
# the absolute label unchanged; only KOSDAQ switches.
LABEL_MODES=("absolute_return","excess_vs_cohort_mean")
LABEL_MODE_BY_COHORT={"KOSPI":"absolute_return","KOSDAQ":"excess_vs_cohort_mean",
                      "US":"absolute_return","TIGER_ETF":"absolute_return"}
DEFAULT_LABEL_MODE="absolute_return"
# A basket mean needs a real cross-section. Groups smaller than this keep no
# excess label (they are dropped from that cohort's modelling, never given a
# substitute label).
MIN_EXCESS_GROUP_N=10
# Exploration safety net for the excess-return label: when no model clears the
# full development selection rule, still report one development-selected model so
# the new label can be measured. It is labelled as an exploration diagnostic, it
# never claims edge, and it cannot turn into a buy policy (direction3 is
# diagnostic-only; every buy decision comes from the legacy ladder).
EXCESS_LABEL_DIAGNOSTIC_FALLBACK=True
EXCESS_LABEL_FALLBACK_STATUS="DIAGNOSTIC_EXCESS_LABEL_EXPLORATION_NOT_A_SELECTION"
SPREAD_CUTS=(.05,.10,.15); SIDEWAYS_MAX=(.25,.35,.45); EV_FLOORS=(0.0,.10,.20)
BASE_FEATURES=("ret5","ret20","ret60","ret120","atr_pct","vol20_pct",
          "rsi","ma20_gap","ma60_gap","ma200_gap","turnover_ratio",
          "volume_flow5","gap_pct","intraday_strength20","xs_ret20",
          "xs_ret60","xs_volatility20","xs_turnover20")
MACRO_FEATURES=("vix_level","vix_change_5d_pct","vix_expanding_percentile","usdkrw_change_20d_pct",
 "sp500_return_20d_pct","sox_return_20d_pct","market_index_ma20_gap_pct","market_index_ma60_gap_pct",
 "market_index_ma200_gap_pct","market_index_return_5d_pct","market_index_return_20d_pct",
 "market_index_return_60d_pct","market_index_volatility_20d_pct","market_volatility_ratio",
 "market_drawdown_120_pct","market_risk_off","market_risk_on")
RELATIVE_FEATURES=("excess_return_20d_pct","excess_return_60d_pct","excess_return_120d_pct")
FLOW_FEATURES=("foreign_flow_5d_to_turnover","foreign_flow_20d_to_turnover",
 "institution_flow_5d_to_turnover","institution_flow_20d_to_turnover","foreign_buy_streak",
 "institution_buy_streak","flow_direction_agreement","flow_data_available")
FEATURES=BASE_FEATURES+MACRO_FEATURES+RELATIVE_FEATURES+FLOW_FEATURES
CENTERS={"ret5":0,"ret20":0,"ret60":0,"ret120":0,"atr_pct":3,"vol20_pct":3,
 "rsi":50,"ma20_gap":0,"ma60_gap":0,"ma200_gap":0,"turnover_ratio":1,
 "volume_flow5":0,"gap_pct":0,"intraday_strength20":.5,"xs_ret20":.5,
 "xs_ret60":.5,"xs_volatility20":.5,"xs_turnover20":.5,
 "vix_level":20,"vix_change_5d_pct":0,"vix_expanding_percentile":.5,"usdkrw_change_20d_pct":0,
 "sp500_return_20d_pct":0,"sox_return_20d_pct":0,"market_index_ma20_gap_pct":0,"market_index_ma60_gap_pct":0,
 "market_index_ma200_gap_pct":0,"market_index_return_5d_pct":0,"market_index_return_20d_pct":0,
 "market_index_return_60d_pct":0,"market_index_volatility_20d_pct":25,"market_volatility_ratio":1,
 "market_drawdown_120_pct":0,"market_risk_off":0,"market_risk_on":0,
 "excess_return_20d_pct":0,"excess_return_60d_pct":0,"excess_return_120d_pct":0,
 "foreign_flow_5d_to_turnover":0,"foreign_flow_20d_to_turnover":0,
 "institution_flow_5d_to_turnover":0,"institution_flow_20d_to_turnover":0,
 "foreign_buy_streak":0,"institution_buy_streak":0,"flow_direction_agreement":0,"flow_data_available":0}
SCALES={"ret5":10,"ret20":20,"ret60":35,"ret120":55,"atr_pct":3,"vol20_pct":5,
 "rsi":25,"ma20_gap":10,"ma60_gap":18,"ma200_gap":30,"turnover_ratio":2,
 "volume_flow5":1,"gap_pct":5,"intraday_strength20":.5,"xs_ret20":.5,
 "xs_ret60":.5,"xs_volatility20":.5,"xs_turnover20":.5,
 "vix_level":20,"vix_change_5d_pct":20,"vix_expanding_percentile":.5,"usdkrw_change_20d_pct":5,
 "sp500_return_20d_pct":10,"sox_return_20d_pct":15,"market_index_ma20_gap_pct":10,"market_index_ma60_gap_pct":18,
 "market_index_ma200_gap_pct":30,"market_index_return_5d_pct":8,"market_index_return_20d_pct":15,
 "market_index_return_60d_pct":25,"market_index_volatility_20d_pct":30,"market_volatility_ratio":1,
 "market_drawdown_120_pct":30,"market_risk_off":1,"market_risk_on":1,
 "excess_return_20d_pct":20,"excess_return_60d_pct":35,"excess_return_120d_pct":55,
 "foreign_flow_5d_to_turnover":2,"foreign_flow_20d_to_turnover":4,
 "institution_flow_5d_to_turnover":2,"institution_flow_20d_to_turnover":4,
 "foreign_buy_streak":10,"institution_buy_streak":10,"flow_direction_agreement":1,"flow_data_available":1}

def direction_label(net_return_pct,theta_pct=.5):
    value=float(net_return_pct)
    if not np.isfinite(value) or theta_pct<=0:raise ValueError("finite return and positive threshold required")
    return "UP" if value>theta_pct else "DOWN" if value< -theta_pct else "SIDEWAYS"

def _ticker_set():
    universe=get_universe(include_us=True)
    # Only the 300 ordinary shares train the expanded equity pool. ETFs remain
    # in the 63-instrument evaluation and SOXL remains in the US cohort.
    rows={"KOSPI":[x for x in universe.get("KR",[]) if x[0] not in {e[0] for e in MANDATORY_KR}],
          "KOSDAQ":list(universe.get("KQ",[])),
          "US":[x for x in universe.get("US",[]) if x[0] not in {e[0] for e in MANDATORY_US}]}
    # Include any frozen benchmark ticker absent from a refreshed ranking feed;
    # this preserves audit coverage while recording an actual load failure.
    for cohort,key,market in (("KOSPI","KOSPI","KR"),("KOSDAQ","KOSDAQ","KQ")):
        present={str(x[0]) for x in rows[cohort]}
        rows[cohort].extend((code,code,market,"frozen audit benchmark")
                            for code in AUDIT_63_BENCHMARK[key] if code not in present)
    rows={c:[(ticker,name,market,sector) for ticker,name,market,sector in items] for c,items in rows.items()}
    rows["universe_sources"]={k:universe.get(k+"_source") for k in ("KR","KQ","US")}
    rows["universe_errors"]={k:universe.get(k+"_error") for k in ("KR","KQ") if universe.get(k+"_error")}
    return rows

def korean_flow_targets(universe):
    """Unique KOSPI/KOSDAQ common stocks only; omit ETFs, ETNs and US rows."""
    excluded={str(code).zfill(6) for code,_,_ in MANDATORY_KR}
    targets=[];seen=set()
    for cohort,market in (("KOSPI","KR"),("KOSDAQ","KQ")):
        for item in universe.get(cohort,[]):
            ticker,name=item[0],item[1]
            code=str(ticker).strip().zfill(6)
            label=str(name).upper()
            if (not re.fullmatch(r"\d{6}",code) or code in excluded or code in seen or
                re.search(r"ETF|ETN|SPAC|스팩|우선주|우$|우B$|우C$|우[0-9]",label)):
                continue
            targets.append({"ticker":code,"name":str(name),"cohort":cohort,"market":market})
            seen.add(code)
    return targets

def _label_mode(cohort):
    """Direction label mode for a cohort (see LABEL_MODE_BY_COHORT)."""
    return LABEL_MODE_BY_COHORT.get(str(cohort),DEFAULT_LABEL_MODE)

def _label_net(row):
    """Net return that the direction label is defined on for this row.

    Equals ``net_pct`` for absolute-return cohorts, so their labels, targets and
    metrics stay bit-for-bit what they were before the excess-label option.
    """
    value=row.get("net_label_pct")
    return float(row["net_pct"]) if value is None else float(value)

def _attach_label_returns(rows,min_group_n=MIN_EXCESS_GROUP_N):
    """Attach ``net_label_pct`` for every row and report the label mode used.

    For ``excess_vs_cohort_mean`` cohorts the basket benchmark is the mean of
    ``net_pct`` over rows that share cohort, signal date and horizon. Those rows
    also share one exit date, so the benchmark uses only prices a live system
    already observes when the member labels mature. Nothing earlier in the
    history is touched and the existing transaction cost stays inside both the
    row and the basket mean (it cancels in the difference).
    """
    finite=lambda row:bool(np.isfinite(float(row.get("net_pct",np.nan))))
    groups=defaultdict(list)
    for i,row in enumerate(rows):
        if finite(row):groups[(row["cohort"],row["signal_date"],int(row["horizon"]))].append(i)
    report={}
    for cohort in sorted({row["cohort"] for row in rows}):
        mode=_label_mode(cohort)
        sizes=[];means=[];excluded=0;labelled=0
        for (c,_day,_horizon),ix in groups.items():
            if c!=cohort:continue
            sizes.append(len(ix))
            if mode!="excess_vs_cohort_mean":continue
            values=np.asarray([float(rows[i]["net_pct"]) for i in ix],dtype=float)
            if len(ix)<int(min_group_n):
                for i in ix:rows[i]["net_label_pct"]=float("nan");excluded+=1
                continue
            mean=float(values.mean());means.append(mean)
            for i in ix:
                row=rows[i]
                row["cohort_mean_net_pct"]=mean
                row["cohort_excess_group_n"]=len(ix)
                row["net_label_pct"]=float(row["net_pct"])-mean
                labelled+=1
        for row in rows:
            if row["cohort"]!=cohort:continue
            if mode!="excess_vs_cohort_mean" or "net_label_pct" not in row:
                # Absolute cohorts (and rows outside any basket) keep the exact
                # net return value, so their pipeline is unchanged.
                row["net_label_pct"]=float(row["net_pct"])
        entry={"mode":mode,"groups_n":len(sizes),
               "group_size_median":float(np.median(sizes)) if sizes else None,
               "group_size_p10":float(np.percentile(sizes,10)) if sizes else None,
               "group_size_max":int(max(sizes)) if sizes else None,
               "min_group_n":int(min_group_n),"rows_labelled_n":labelled,
               "rows_without_basket_n":excluded}
        if mode=="excess_vs_cohort_mean":
            entry.update({"benchmark":"equal-weighted mean net return of the same cohort, signal date and horizon (same exit date)",
                "definition":"net_label_pct = net_pct - cohort/signal-date/horizon basket mean net_pct; the existing transaction cost is inside both terms and cancels",
                "label_available_date":"unchanged: the horizon exit close, identical to the basket members' own reveal date",
                "basket_mean_p10_p50_p90_pct":(np.percentile(means,[10,50,90]).tolist() if means else None)})
        report[cohort]=entry
    return report

def _metric_net(row,net_key):
    """Return used by ranking metrics; falls back to the raw net return."""
    value=row.get(net_key)
    return float(row["net_pct"]) if value is None else float(value)

def _metrics_net_key(rows):
    """Ranking metric key: the label's own return when a cohort uses excess."""
    for row in rows or []:
        if _label_mode(row.get("cohort",""))=="excess_vs_cohort_mean":
            return "net_label_pct"
    return "net_pct"

def _development_accuracy(development_result):
    """Three-way accuracy of the development-calibrated decision rule."""
    metrics=(development_result or {}).get("metrics") or {}
    decision=metrics.get("class_balanced_decision") or {}
    value=decision.get("accuracy_pct")
    return float(value) if value is not None else None

def _beats_causal_prior_on_development(development_result):
    """Model must beat the causal prior on development accuracy AND on IC sign."""
    metrics=(development_result or {}).get("metrics") or {}
    if not (metrics.get("datewise_spearman_mean") or 0)>0:
        return False
    accuracy=_development_accuracy(development_result)
    causal=(metrics.get("probability_skill") or {}).get("causal_prior_accuracy_pct")
    if accuracy is None or causal is None:
        return False
    return accuracy>=float(causal)

def _edge_label_status(model_selected,chosen_ic):
    if not model_selected:
        return "NO_MODEL_WITH_NONZERO_DEVELOPMENT_N"
    if chosen_ic is not None and np.isfinite(chosen_ic) and float(chosen_ic)>=MIN_DEVELOPMENT_IC:
        return "DEVELOPMENT_EDGE_SELECTED"
    return "DIAGNOSTIC_MODEL_SELECTED_NO_DEVELOPMENT_EDGE"

def _collect_extended_pool_flows(universe,flow_frames,*,flow_source_coverage=None,budget_seconds=None,workers=None,
                                 flow_fetcher=None,progress=None,clock=None):
    """Extend the audit's existing 63 flow frames to all Korean model stocks.

    New symbols are collected in bounded concurrent waves. Existing benchmark
    frames are reused verbatim, and unfinished names remain absent so feature
    construction keeps their flow values missing rather than fabricating zeros.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    if flow_fetcher is None:flow_fetcher=get_historical_investor_flows
    if clock is None:clock=time.monotonic
    budget=float(os.environ.get("FLOW_BUDGET_SECONDS","3000") if budget_seconds is None else budget_seconds)
    worker_count=max(1,int(os.environ.get("FLOW_WORKERS","12") if workers is None else workers))
    budget=max(0.0,budget)
    targets=korean_flow_targets(universe)
    started=clock();known={str(key).zfill(6):value for key,value in flow_frames.items()}
    # Preserve the caller's dictionary and its exact existing DataFrame objects.
    for code,value in list(flow_frames.items()):known[str(code).zfill(6)]=value
    reused={x["ticker"] for x in targets if x["ticker"] in known and
            known[x["ticker"]] is not None and not known[x["ticker"]].empty}
    pending=[x for x in targets if x["ticker"] not in reused]
    source_coverage=flow_source_coverage or {}
    attempted=[];available_new=set();failures=[];rows_new=0;requests_n=0
    ticker_results={};completed_count=0
    retry_log=[];not_attempted=[];reserve_seconds=min(180.0,max(0.0,budget*.06))
    index=0
    while index<len(pending):
        elapsed=max(0.0,clock()-started)
        if budget-elapsed<=reserve_seconds:
            not_attempted.extend({**x,"reason":"flow time budget exhausted before request"} for x in pending[index:])
            break
        batch=pending[index:index+worker_count];index+=len(batch)
        attempted.extend(x["ticker"] for x in batch)
        with ThreadPoolExecutor(max_workers=min(worker_count,len(batch))) as executor:
            futures={executor.submit(flow_fetcher,item["ticker"],start_date="2015-12-01",
                                    end_date=MIN_END_DATE):item for item in batch}
            for future in as_completed(futures):
                completed_count+=1
                item=futures[future];code=item["ticker"]
                try:
                    frame,source=future.result()
                    if frame is None or frame.empty:
                        raise ValueError("flow collector returned an empty frame")
                    diag=frame.attrs.get("collection_diagnostics",{})
                    frame.attrs.setdefault("collection_diagnostics",diag)
                    flow_frames[code]=frame;known[code]=frame;available_new.add(code)
                    rows_new+=len(frame);requests_n+=int(diag.get("requests_n",0) or 0)
                    retry_log.extend(diag.get("retry_log",[]) or [])
                    ticker_results[code]={**diag,"ticker":code,"status":"ok","source":source,
                        "available":True,"rows_n":len(frame)}
                    if progress:progress(item,len(targets),completed_count,len(frame),int(diag.get("requests_n",0) or 0),"ok")
                except Exception as exc:
                    diag=getattr(exc,"diagnostics",{}) or {}
                    requests_n+=int(diag.get("requests_n",0) or 0)
                    retry_log.extend(diag.get("retry_log",[]) or [])
                    failure={**item,"reason":str(exc)[:500],"requests_n":int(diag.get("requests_n",0) or 0)}
                    failures.append(failure)
                    ticker_results[code]={**diag,**failure,"status":"unavailable","available":False}
                    if progress:progress(item,len(targets),completed_count,0,int(diag.get("requests_n",0) or 0),"unavailable")
        if index<len(pending) and clock()-started>=budget-reserve_seconds:
            not_attempted.extend({**x,"reason":"flow time budget exhausted before request"} for x in pending[index:])
            break
    elapsed=max(0.0,clock()-started)
    available={x["ticker"] for x in targets if x["ticker"] in known and
               known[x["ticker"]] is not None and not known[x["ticker"]].empty}
    reused_by_cohort={c:{x["ticker"] for x in targets if x["cohort"]==c and x["ticker"] in reused}
                      for c in ("KOSPI","KOSDAQ")}
    attempted_by_cohort={c:{x["ticker"] for x in targets if x["cohort"]==c and x["ticker"] in set(attempted)}
                         for c in ("KOSPI","KOSDAQ")}
    available_by_cohort={c:{x["ticker"] for x in targets if x["cohort"]==c and x["ticker"] in available}
                         for c in ("KOSPI","KOSDAQ")}
    target_by_cohort={c:[x for x in targets if x["cohort"]==c] for c in ("KOSPI","KOSDAQ")}
    coverage={}
    for c,items in target_by_cohort.items():
        available_set=available_by_cohort[c]
        coverage[c]={"requested_symbols":len(items),"preexisting_reused_symbols":len(reused_by_cohort[c]),
          "attempted_symbols":len(attempted_by_cohort[c]),"available_symbols":len(available_set),
          "coverage_pct":round(100*len(available_set)/len(items),3) if items else None,
          "rows_n":sum(len(known[t]) for t in available_set)}
    not_attempted_codes={x["ticker"] for x in not_attempted}
    by_ticker={}
    for item in targets:
        code=item["ticker"]
        if code in ticker_results:
            by_ticker[code]=ticker_results[code]
        elif code in reused:
            frame=known[code];diag=frame.attrs.get("collection_diagnostics",{})
            by_ticker[code]={**source_coverage.get(code,{}),**diag,"ticker":code,
                "status":"reused_existing_63_flow","source":source_coverage.get(code,{}).get("source"),
                "attempted":False,"available":True,"rows_n":len(frame),
                "first_date":str(pd.Timestamp(frame.index.min()).date()),
                "last_date":str(pd.Timestamp(frame.index.max()).date())}
        elif code in not_attempted_codes:
            by_ticker[code]={**item,"status":"not_attempted_budget","attempted":False,
                "available":False,"rows_n":0,"requests_n":0,"reason":"flow time budget exhausted before request"}
    report={"source":"Naver mobile stock trend API; unauthenticated; no KRX login/paid API",
      "requested_start":"2015-12-01","requested_end":MIN_END_DATE,
      "extended_296_pool_attempted":bool(attempted),
      "requested_symbols":len(targets),"preexisting_reused_symbols":len(reused),
      "attempted_symbols":len(attempted),"available_symbols":len(available),
      "newly_available_symbols":len(available_new),"total_requests_n":requests_n,
      "rows_n":sum(len(known[t]) for t in available),"new_rows_n":rows_new,
      "budget_seconds":budget,"budget_used_seconds":round(elapsed,3),
      "budget_exhausted":bool(not_attempted or elapsed>=budget),
      "workers":worker_count,"attempted_tickers":attempted,
      "failures":failures,"not_attempted":not_attempted,"by_ticker":by_ticker,"retry_log":retry_log,
      "coverage_by_cohort":coverage}
    return report

def _audit_symbols(universe):
    return {"KOSPI":set(AUDIT_63_BENCHMARK["KOSPI"]),
            "KOSDAQ":set(AUDIT_63_BENCHMARK["KOSDAQ"]),
            "US":set(AUDIT_63_BENCHMARK["SOXL"]),
            "TIGER_ETF":set(AUDIT_63_BENCHMARK["TIGER_ETF"])}

def _mandatory_market_items(items, market):
    """Normalize universe.py's (ticker, name, reason) entries for price loading."""
    for ticker, name, _reason in items:
        yield ticker, name, market

def _make_rows(frame,ticker,market,cohort):
    x=compute(frame); rows=[]; close=x.Close.astype(float); volume=x.Volume.astype(float)
    turnover=close*volume; high=x.High.astype(float); low=x.Low.astype(float); op=x.Open.astype(float)
    rets=close.pct_change(); atr=x.ATR.astype(float); vol20=rets.rolling(20,min_periods=20).std(ddof=0)*100
    tv20=turnover.rolling(20,min_periods=20).mean()
    for i in range(200,len(x)-20,SIGNAL_STEP):
        day=pd.Timestamp(x.index[i]).date().isoformat()
        if day<SAMPLE_START or day>=HOLDOUT_END:continue
        entry=float(op.iloc[i+1]); px=float(close.iloc[i]);
        if not np.isfinite(entry) or entry<=0 or abs(entry/px-1)>.04:continue
        r=x.iloc[i]; vals={
          "ret5":100*(px/float(close.iloc[i-5])-1),"ret20":100*(px/float(close.iloc[i-20])-1),
          "ret60":100*(px/float(close.iloc[i-60])-1),"ret120":100*(px/float(close.iloc[i-120])-1),
          "atr_pct":100*float(atr.iloc[i])/px,"vol20_pct":float(vol20.iloc[i]),
          "rsi":float(r.RSI),"ma20_gap":100*(px/float(r.MA20)-1),
          "ma60_gap":100*(px/float(r.MA60)-1),"ma200_gap":100*(px/float(r.MA200)-1),
          "turnover_ratio":float(turnover.iloc[i]/tv20.iloc[i]) if tv20.iloc[i]>0 else 1.,
          "volume_flow5":float(np.sign(rets.iloc[max(0,i-4):i+1]).dot(volume.iloc[max(0,i-4):i+1]))/max(1,float(volume.iloc[max(0,i-4):i+1].sum())),
          "gap_pct":100*(float(op.iloc[i])/float(close.iloc[i-1])-1),
          "intraday_strength20":float(((close-low)/(high-low).replace(0,np.nan)).iloc[max(0,i-19):i+1].mean())}
        if not all(np.isfinite(vals[k]) for k in FEATURES[:14]):continue
        for h in HORIZONS:
            exit_index=i+h
            if exit_index>=len(x):continue
            raw=100*(float(close.iloc[exit_index])/entry-1)
            net=raw-COST_PCT
            touch="NEITHER";target=entry*1.005;stop=entry*.995
            for j in range(i+1,exit_index+1):
                hit_stop=float(low.iloc[j])<=stop;hit_target=float(high.iloc[j])>=target
                if hit_stop:touch="STOP_FIRST";break
                if hit_target:touch="TARGET_FIRST";break
            row={"cohort":cohort,"ticker":ticker,"market":market,"signal_date":day,
                 "entry_date":pd.Timestamp(x.index[i+1]).date().isoformat(),
                 "exit_date":pd.Timestamp(x.index[exit_index]).date().isoformat(),
                 "label_available_date":pd.Timestamp(x.index[exit_index]).date().isoformat(),
                 "horizon":h,"net_pct":net,"target_touch_theta_0.5":touch,**vals}
            for theta in THETAS:row[f"label_{theta}"]=direction_label(net,theta)
            # Volatility is measured through the signal-day close only. The
            # cohort k is selected from development labels, then frozen for OOS.
            row["volatility_unit_pct"]=float(vol20.iloc[i])*float(np.sqrt(h/20))
            rows.append(row)
    return rows

def _rank_features(rows):
    byday=defaultdict(list)
    for i,r in enumerate(rows):byday[(r["cohort"],r["signal_date"],r["horizon"])].append(i)
    for (cohort,day,horizon),ix in byday.items():
        for key,outkey in (("ret20","xs_ret20"),("ret60","xs_ret60"),
                           ("vol20_pct","xs_volatility20"),("turnover_ratio","xs_turnover20")):
            values=np.asarray([rows[j][key] for j in ix],float)
            valid=np.isfinite(values);n=int(valid.sum())
            ranks=np.full(len(ix),np.nan)
            if n>1:ranks[valid]=pd.Series(values[valid]).rank(pct=True,method="average").to_numpy()
            for a,j in enumerate(ix):
                rows[j][outkey]=float(ranks[a]) if np.isfinite(ranks[a]) else CENTERS[outkey]
                rows[j][f"{outkey}_sample_n"]=n
        for j in ix:rows[j]["xs_count"]=len(ix)
    return rows

def _attach_context_features(rows,frames_by_cohort,macro_frames,flow_frames):
    """Attach close-timestamped macro, relative-strength and observed flow data."""
    direct=MACRO_FEATURES+FLOW_FEATURES
    excess_map={"stock_excess_return_20d_pct":"excess_return_20d_pct",
                "stock_excess_return_60d_pct":"excess_return_60d_pct",
                "stock_excess_return_120d_pct":"excess_return_120d_pct"}
    for cohort,frames in frames_by_cohort.items():
        if not frames:continue
        cohort_flows={ticker:flow_frames[ticker] for ticker in frames if ticker in flow_frames}
        context=build_market_context(frames,macro_frames,cohort,cohort_flows)
        for row in rows:
            if row["cohort"]!=cohort:continue
            values=context["features"].get(row["ticker"],{}).get(row["signal_date"],{})
            for key in direct:
                value=values.get(key,np.nan)
                # A measured zero explicitly means no flow coverage. Other
                # missing flow values stay NaN and are never imputed as zero.
                if key=="flow_data_available" and not np.isfinite(value):value=0.0
                row[key]=float(value) if value is not None else np.nan
            for source,target in excess_map.items():row[target]=values.get(source,np.nan)
            row["market_regime"]=values.get("market_regime","UNKNOWN")
    return rows

def _active_features(rows):
    active=[]
    for key in FEATURES:
        values=np.asarray([float(r.get(key,CENTERS[key])) for r in rows],dtype=float)
        values=values[np.isfinite(values)]
        if len(values) and np.std(values)>1e-12:active.append(key)
    return tuple(active)

def _features(rows,active=None):
    active=FEATURES if active is None else active
    def scaled(row,key):
        value=float(row.get(key,CENTERS[key]))
        return float(np.clip((value-CENTERS[key])/SCALES[key],-5,5)) if np.isfinite(value) else np.nan
    return np.asarray([[scaled(r,k) for k in active] for r in rows],dtype=np.float32)

def _refit_anchor(signal_date):
    """Calendar anchor of the refit block that a signal belongs to.

    Blocks are fixed-length in months counted from year 0, so REFIT_MONTH=24
    reproduces the historical two-year snapshots (2024-2025 -> 2024-01-01).
    Labels used for a block must be observed strictly before this anchor, so a
    shorter cadence can only see the past and never the block itself.
    """
    year=int(signal_date[:4]); month=int(signal_date[5:7])
    step=max(1,int(REFIT_MONTH))
    absolute=year*12+(month-1)
    first=(absolute//step)*step
    return f"{first//12:04d}-{first%12+1:02d}-01"

class BandRegressionModel:
    """Ridge on the net percentage return, mapped through the label band.

    The audit's SIDEWAYS band is ``k * signal-day volatility`` and is known at
    the signal close, so a predicted return ``mu`` with residual scale ``sigma``
    gives P(UP)=1-Phi((band-mu)/sigma) then P(DOWN)=Phi((-band-mu)/sigma).
    Uses return magnitudes instead of only the discrete class label.

    ``winsor`` clips the training target at its own lower/upper percentiles (fit
    on the same past-only rows) and ``normalise`` divides the target by the
    signal-day volatility unit, so a single extreme move cannot dominate the
    fit; ``scale="mad"`` replaces the residual std with a robust MAD estimate.
    """
    classes_=np.asarray([0,1,2])
    def __init__(self,alpha=1.0,winsor=None,normalise=False,scale="std",huber=False):
        self.alpha=float(alpha);self.winsor=winsor;self.normalise=bool(normalise)
        self.scale=scale;self.huber=bool(huber)
        from sklearn.linear_model import HuberRegressor
        estimator=(HuberRegressor(epsilon=1.35,alpha=max(self.alpha,1e-4),max_iter=500)
                   if self.huber else Ridge(alpha=self.alpha))
        self.routine=make_pipeline(SimpleImputer(strategy="constant",fill_value=0),
                                   StandardScaler(),estimator)
    def fit(self,X,y):
        y=np.asarray(y,dtype=float)
        if self.winsor is not None:
            lower,upper=np.nanpercentile(y,self.winsor)
            y=np.clip(y,lower,upper);self.winsor_bounds=(float(lower),float(upper))
        try:
            self.routine.fit(X,y)
        except Exception:
            self.routine=make_pipeline(SimpleImputer(strategy="constant",fill_value=0),
                                       StandardScaler(),Ridge(alpha=max(self.alpha,1e-6))).fit(X,y)
        residual=y-self.routine.predict(X)
        if self.scale=="mad":
            mad=float(np.median(np.abs(residual-np.median(residual))))*1.4826
            self.sigma=float(max(mad,np.std(residual)*.25,1e-6))
        else:
            self.sigma=float(max(np.std(residual),1e-6))
        return self
    def predict_proba(self,X,band=None):
        from scipy.stats import norm
        mu=np.asarray(self.routine.predict(X),dtype=float)
        band=np.abs(np.asarray(band if band is not None else np.ones(len(mu)),dtype=float))
        up=1-norm.cdf((band-mu)/self.sigma)
        down=norm.cdf((-band-mu)/self.sigma)
        probs=np.column_stack([down,np.clip(1-up-down,0.,1.),up])
        probs=np.clip(probs,1e-6,1.)
        return probs/probs.sum(axis=1,keepdims=True)

def _band_pct(row):
    """SIDEWAYS band in net-return percentage points, known at the signal close."""
    return abs(float(row.get("label_vol_band_pct",0.) or 0.))

def _volatility_unit(row):
    value=float(row.get("volatility_unit_pct",0.) or 0.)
    return value if np.isfinite(value) and value>1e-9 else 1e-9

def _band_z(row):
    """SIDEWAYS band in volatility units (== cohort k) for normalised targets."""
    return _band_pct(row)/_volatility_unit(row)

def _model(kind):
    if kind=="band_regression":
        return BandRegressionModel()
    if kind=="band_winsor":
        return BandRegressionModel(winsor=(1.,99.),scale="mad")
    if kind=="band_huber":
        return BandRegressionModel(huber=True,scale="mad")
    if kind=="band_z":
        return BandRegressionModel(normalise=True,winsor=(1.,99.),scale="mad")
    if kind=="multinomial_logistic":
        return make_pipeline(SimpleImputer(strategy="constant",fill_value=0),StandardScaler(),
             LogisticRegression(C=.15,max_iter=300,class_weight=None,random_state=3610))
    return make_pipeline(SimpleImputer(strategy="constant",fill_value=0),
            RandomForestClassifier(n_estimators=80,max_depth=5,min_samples_leaf=20,
            class_weight=None,random_state=3610,n_jobs=1))

def _freeze_volatility_labels(rows):
    """Choose a cohort-wide k using only labels mature before TRAIN_END."""
    result={}; names=("DOWN","SIDEWAYS","UP")
    for cohort in sorted({r["cohort"] for r in rows}):
        mode=_label_mode(cohort)
        dev=[r for r in rows if r["cohort"]==cohort and r["label_available_date"]<TRAIN_END
             and np.isfinite(r.get("volatility_unit_pct",np.nan)) and r["volatility_unit_pct"]>0
             and np.isfinite(_label_net(r))]
        z=np.asarray([abs(_label_net(r))/float(r["volatility_unit_pct"]) for r in dev])
        k=float(np.quantile(z,SIDEWAYS_TARGET_SHARE)) if len(z) else None
        labels=[]
        if k is not None:
            for row in rows:
                if row["cohort"]!=cohort:continue
                band=k*float(row.get("volatility_unit_pct",0.0))
                value=_label_net(row)
                # A row whose basket was too thin carries no label return and is
                # left unlabelled rather than given a substitute one; it is then
                # dropped from that cohort's modelling.
                if band>0 and np.isfinite(value):
                    # The band thresholds the same return the cohort's label mode
                    # defines, so a stock beat/missed its basket, not just the tape.
                    row["label_vol"]=direction_label(value,band);row["label_vol_band_pct"]=band
            labels=[r["label_vol"] for r in dev]
        counts={name:labels.count(name) for name in names}
        result[cohort]={"k":k,"label_mode":mode,"development_n":len(dev),"development_class_counts":counts,
          "development_class_share_pct":{n:round(100*counts[n]/len(dev),3) if dev else None for n in names},
          "definition":("SIDEWAYS iff abs(net return after existing costs) <= k * signal-day 20d realized daily volatility * sqrt(h/20); UP/DOWN outside band"
            if mode=="absolute_return" else
            "SIDEWAYS iff abs(net return after existing costs minus the same cohort/signal-date/horizon basket mean net return) <= k * signal-day 20d realized daily volatility * sqrt(h/20); UP/DOWN outside band"),
          "label_return":"net_pct" if mode=="absolute_return" else "net_pct - cohort basket mean net_pct",
          "fit_scope":"development labels available strictly before TRAIN_END; k frozen for validation/diagnostic"}
    return result

def _apply_temperature(probs,temperature):
    p=np.clip(np.asarray(probs,dtype=float),1e-9,1.)
    logits=np.log(p)/max(float(temperature),1e-6)
    logits-=logits.max(axis=1,keepdims=True)
    exp=np.exp(logits)
    return exp/exp.sum(axis=1,keepdims=True)

def _fit_temperature(rows,probs,label_key="label_vol"):
    names=("DOWN","SIDEWAYS","UP")
    y=np.asarray([names.index(r[label_key]) for r in rows],dtype=int)
    p=np.asarray(probs,dtype=float)
    if len(y)<100:return {"temperature":1.0,"fit_n":len(y),"status":"INSUFFICIENT_DEV_OOF_FOR_CALIBRATION"}
    def objective(log_t):
        calibrated=_apply_temperature(p,np.exp(log_t))
        return float(log_loss(y,calibrated,labels=[0,1,2]))
    fit=minimize_scalar(objective,bounds=(-1.5,1.5),method="bounded",options={"xatol":1e-5})
    return {"temperature":float(np.exp(fit.x)),"fit_n":len(y),"status":"DEVELOPMENT_OOF_TEMPERATURE_SCALING",
            "development_oof_logloss_before":objective(0.),"development_oof_logloss_after":float(fit.fun)}

def _choose_development_k(rows):
    """Compatibility alias for callers/tests; returns frozen cohort settings."""
    return _freeze_volatility_labels(rows)

def _causal_class_priors(all_train,eval_rows,window_days=PRIOR_SHRINKAGE_WINDOW_DAYS):
    """Per-signal-date class prior from labels already observable at that date.

    Only labels with ``label_available_date`` strictly before the signal date are
    counted, so the prior can be computed live. The window is a choice made on
    development rows only; when the window holds fewer than MIN_PRIOR_ROWS
    labels the expanding history before the signal date is used, and only when
    that is also too small does the prior fall back to uniform.
    """
    names=("DOWN","SIDEWAYS","UP")
    labelled=[r for r in all_train if r.get("label_vol") in names]
    if not labelled or not eval_rows:
        return np.full((len(eval_rows),3),1/3,dtype=float)
    dates=np.asarray([r["label_available_date"] for r in labelled])
    order=np.argsort(dates,kind="mergesort");dates=dates[order]
    onehot=np.zeros((len(labelled),3),dtype=float)
    for i,row in enumerate(labelled):
        onehot[i,names.index(row["label_vol"])]=1.
    cumulative=np.vstack([np.zeros(3),np.cumsum(onehot[order],axis=0)])
    cache={};out=np.full((len(eval_rows),3),1/3,dtype=float)
    for i,row in enumerate(eval_rows):
        signal=str(row["signal_date"])
        if signal not in cache:
            floor=(date.fromisoformat(signal)-timedelta(days=int(window_days))).isoformat()
            low=int(np.searchsorted(dates,floor,"left"));high=int(np.searchsorted(dates,signal,"left"))
            counts=cumulative[high]-cumulative[low]
            if counts.sum()<MIN_PRIOR_ROWS:
                counts=cumulative[high]
            cache[signal]=counts/counts.sum() if counts.sum()>0 else None
        value=cache[signal]
        if value is not None:
            out[i]=value
    return out

def _multiclass_brier(rows,probs,label_key="label_vol"):
    names=("DOWN","SIDEWAYS","UP")
    y=np.asarray([names.index(r[label_key]) for r in rows],dtype=int)
    actual=np.zeros((len(y),3),dtype=float);actual[np.arange(len(y)),y]=1.
    return float(np.mean(np.sum((np.asarray(probs,dtype=float)-actual)**2,axis=1)))

def _fit_prior_mix(rows,probs,priors):
    """Weight on the causal prior that minimises development out-of-fold Brier."""
    if len(rows)<100:return 0.
    def objective(weight):
        weight=float(min(max(weight,0.),1.))
        return _multiclass_brier(rows,(1-weight)*np.asarray(probs,dtype=float)+weight*np.asarray(priors,dtype=float))
    fit=minimize_scalar(objective,bounds=(0.,1.),method="bounded",options={"xatol":1e-4})
    return float(min(max(fit.x,0.),1.))

def _apply_prior_mix(probs,priors,weight):
    weight=float(min(max(weight or 0.,0.),1.))
    if weight<=0:
        return np.asarray(probs,dtype=float)
    mixed=(1-weight)*np.asarray(probs,dtype=float)+weight*np.asarray(priors,dtype=float)
    return mixed/mixed.sum(axis=1,keepdims=True)

def _prior_corrected_decisions(probs,priors,tau):
    """Argmax of p/q^tau: a class-balanced decision under a prior shift.

    tau=0 reproduces the plain probability argmax. A positive tau lowers classes
    the causal prior already expects, so a rare class can still be chosen.
    """
    tau=float(tau or 0.)
    if tau<=0:
        return np.asarray(probs,dtype=float).argmax(axis=1)
    adjusted=np.clip(np.asarray(probs,dtype=float),1e-9,1.)/\
        np.power(np.clip(np.asarray(priors,dtype=float),1e-3,1.),tau)
    return adjusted.argmax(axis=1)

def _fit_decision_tau(rows,probs,priors):
    """Decision threshold chosen on development accuracy only."""
    names=("DOWN","SIDEWAYS","UP")
    y=np.asarray([names.index(r["label_vol"]) for r in rows],dtype=int)
    best=(-1.,0.)
    for tau in DECISION_TAU_GRID:
        accuracy=float(np.mean(_prior_corrected_decisions(probs,priors,tau)==y))
        if accuracy>best[0]+1e-12:best=(accuracy,float(tau))
    return best[1],best[0]

def _fit_calibration(rows,probs,priors):
    """Development-only temperature, prior shrinkage and decision threshold."""
    base=_fit_temperature(rows,probs)
    temperature=float(base.get("temperature",1.0))
    scaled=_apply_temperature(probs,temperature)
    weight=_fit_prior_mix(rows,scaled,priors)
    mixed=_apply_prior_mix(scaled,priors,weight)
    tau,accuracy=_fit_decision_tau(rows,mixed,priors)
    result=dict(base)
    result.update({"prior_mix":float(weight),"prior_window_days":PRIOR_SHRINKAGE_WINDOW_DAYS,
        "decision_tau":float(tau),
        "development_brier_before_prior_mix":_multiclass_brier(rows,scaled),
        "development_brier_after_prior_mix":_multiclass_brier(rows,mixed),
        "development_accuracy_argmax_pct":100*float(np.mean(mixed.argmax(axis=1)==
            np.asarray([("DOWN","SIDEWAYS","UP").index(r["label_vol"]) for r in rows]))),
        "development_accuracy_after_decision_pct":100*float(accuracy),
        "scope":"development out-of-fold predictions only; applied unchanged to validation and diagnostic"})
    if result.get("status")=="INSUFFICIENT_DEV_OOF_FOR_CALIBRATION":
        result.update({"prior_mix":0.,"decision_tau":0.})
    return result

def _ece(y,p,klass):
    confidence=p[:,klass]; actual=(y==klass).astype(float); ece=0.; bins=[]
    for b in range(10):
        lo=b/10; hi=(b+1)/10; mask=(confidence>=lo)&((confidence<hi) if b<9 else (confidence<=hi))
        if mask.any():
            conf=float(confidence[mask].mean()); acc=float(actual[mask].mean()); ece+=mask.sum()/len(y)*abs(conf-acc)
            bins.append({"bin":f"{lo:.1f}-{hi:.1f}","n":int(mask.sum()),"confidence":conf,"observed":acc})
    return {"ece":ece,"bins":bins}

def _metrics(rows,probs,theta=.5,label_key=None,net_key="net_pct"):
    names=("DOWN","SIDEWAYS","UP"); label_key=label_key or f"label_{theta}"; y=np.asarray([names.index(r[label_key]) for r in rows]); pred=probs.argmax(axis=1)
    cm=confusion_matrix(y,pred,labels=[0,1,2]);pr,re,fs,_=precision_recall_fscore_support(y,pred,labels=[0,1,2],zero_division=0)
    briers=[];eces={}
    for j,name in enumerate(names):
        actual=(y==j).astype(float); briers.append(float(np.mean((probs[:,j]-actual)**2))); eces[name]=_ece(y,probs,j)
    class_counts=np.bincount(y,minlength=3);priors=class_counts/max(1,len(y))
    prior_entropy=float(-np.sum(priors[priors>0]*np.log(priors[priors>0])))
    out={"n":len(rows),"label_key":label_key,"ranking_net_key":net_key,"class_distribution":{names[j]:int(class_counts[j]) for j in range(3)},
      "accuracy_pct":100*accuracy_score(y,pred),"majority_class_baseline_accuracy_pct":100*float(class_counts.max()/len(y)) if len(y) else None,
      "balanced_accuracy_pct":100*balanced_accuracy_score(y,pred),
      "macro_f1_pct":100*f1_score(y,pred,average="macro",zero_division=0),"class_metrics":{},
      "confusion_matrix":{"labels":names,"counts":cm.tolist()},"multiclass_log_loss":log_loss(y,probs,labels=[0,1,2]),
      "constant_prior_entropy_nats":prior_entropy,
      "logloss_vs_constant_prior_nats":log_loss(y,probs,labels=[0,1,2])-prior_entropy,
      "class_brier":{n:briers[j] for j,n in enumerate(names)},"mean_class_brier":float(np.mean(briers)),"class_ece":eces}
    touches=[r.get("target_touch_theta_0.5") for r in rows]
    out["target_price_hit_diagnostic"]={"target_first_n":sum(x=="TARGET_FIRST" for x in touches),
      "stop_first_n":sum(x=="STOP_FIRST" for x in touches),"neither_n":sum(x=="NEITHER" for x in touches),
      "target_first_pct":100*sum(x=="TARGET_FIRST" for x in touches)/len(touches) if touches else None,
      "path_timing":"t+1 through t+h; same-bar stop/target resolves stop-first; gross ±0.5% barriers"}
    for j,name in enumerate(names):out["class_metrics"][name]={"precision_pct":100*pr[j],"recall_pct":100*re[j],"f1_pct":100*fs[j]}
    up_score=probs[:,2]; actual_up=(y==2).astype(int)
    out["up_auroc"]=float(roc_auc_score(actual_up,up_score)) if len(np.unique(actual_up))==2 else None
    daily=defaultdict(list)
    for i,r in enumerate(rows):daily[r["signal_date"]].append(i)
    daily_ic=[]
    for indexes in daily.values():
        if len(indexes)<3:continue
        v=spearmanr(up_score[indexes],[_metric_net(rows[i],net_key) for i in indexes]).statistic
        if np.isfinite(v):daily_ic.append(float(v))
    order=np.argsort(up_score,kind="mergesort");k=max(1,int(np.ceil(len(rows)*.15)));top=order[-k:]
    out["datewise_spearman_mean"]=float(np.mean(daily_ic)) if daily_ic else None
    out["datewise_spearman_n_days"]=len(daily_ic)
    out["top15_n"]=len(top);out["top15_win_rate_pct"]=100*float(np.mean([_metric_net(rows[i],net_key)>0 for i in top]))
    out["top15_mean_net_pct"]=float(np.mean([_metric_net(rows[i],net_key) for i in top]))
    return out

def _trade_perf(selected,all_rows,probs,theta=.5):
    net=np.asarray([float(r["net_pct"]) for r in selected],dtype=float)
    pos=float(net[net>0].sum()) if len(net) else 0.; neg=float(-net[net<0].sum()) if len(net) else 0.
    equity=1.;peak=1.;mdd=0.
    for value in net:
        equity*=1+value/100;peak=max(peak,equity);mdd=max(mdd,(peak-equity)/peak)
    active=[];daily=defaultdict(float);used=set();portfolio_trades=[]
    for r in sorted(selected,key=lambda x:(x["entry_date"],x["ticker"])):
        active=[x for x in active if x.get("exit_date",x["label_available_date"])>=r["entry_date"]]
        if len(active)>=5:continue
        active.append(r);portfolio_trades.append(r);used.add(r["ticker"]);daily[r["label_available_date"]]+=float(r["net_pct"])/5/100
    eq=1.;peak=1.;slot_mdd=0.
    for day in sorted(daily):
        eq*=1+daily[day];peak=max(peak,eq);slot_mdd=max(slot_mdd,(peak-eq)/peak)
    down_pred=[i for i,p in enumerate(probs) if int(np.argmax(p))==0]
    side_pred=[i for i,p in enumerate(probs) if int(np.argmax(p))==1]
    return {"n":len(selected),"win_rate_pct":100*float(np.mean(net>0)) if len(net) else None,
      "mean_net_pct":float(net.mean()) if len(net) else None,"profit_factor":pos/neg if neg else (None if not pos else "infinite"),
      "max_drawdown_sequence_pct":100*mdd if len(net) else None,"max_drawdown_5slot_pct":100*slot_mdd if len(net) else None,
      "portfolio_5slot_trade_n":len(portfolio_trades),
      "symbol_coverage_n":len({r["ticker"] for r in selected}),
      "symbol_coverage_pct":100*len({r["ticker"] for r in selected})/len({r["ticker"] for r in all_rows}) if all_rows else 0,
      "coverage_pct":100*len(selected)/len(all_rows) if all_rows else 0,
      "rejected_down_n":len(down_pred),"rejected_down_loss_avoided_pct":-sum(float(all_rows[i]["net_pct"]) for i in down_pred if float(all_rows[i]["net_pct"])<0),
      "rejected_down_correct_pct":100*float(np.mean([all_rows[i][f"label_{theta}"]=="DOWN" for i in down_pred])) if down_pred else None,
      "rejected_sideways_n":len(side_pred),"rejected_sideways_loss_avoided_pct":-sum(float(all_rows[i]["net_pct"]) for i in side_pred if float(all_rows[i]["net_pct"])<0)}

def _class_payoffs(rows):
    return {name:(float(np.mean([r["net_pct"] for r in rows if r["label_vol"]==name]))
                  if any(r["label_vol"]==name for r in rows) else 0.)
            for name in ("DOWN","SIDEWAYS","UP")}

def _policy_table(rows,probs,payoffs):
    up,side,down=probs[:,2],probs[:,1],probs[:,0]
    expected=up*payoffs["UP"]+side*payoffs["SIDEWAYS"]+down*payoffs["DOWN"]
    table={}
    for spread_cut in SPREAD_CUTS:
      for side_max in SIDEWAYS_MAX:
       for ev_floor in EV_FLOORS:
        key=f"spread_ge_{spread_cut:.2f}_side_le_{side_max:.2f}_netEV_ge_{ev_floor:.2f}"
        mask=((up-down)>=spread_cut)&(side<=side_max)&(expected>=ev_floor)
        chosen=[r for r,keep in zip(rows,mask) if keep]
        value=_trade_perf(chosen,rows,probs)
        value.update({"spread_threshold":spread_cut,"sideways_probability_max":side_max,
                      "net_expected_value_floor_pct":ev_floor})
        table[key]=value
    return table

def _causal_prior_baseline(all_train,eval_rows,window_days=CAUSAL_PRIOR_WINDOW_DAYS):
    """Plurality class of labels that were already observable before each block.

    Unlike the majority-class baseline (which is computed from the block's own
    labels and is therefore an oracle), this baseline can be computed in real
    time. If the model cannot beat it, the model adds no usable edge.
    """
    if not eval_rows:return {"n":0,"window_days":window_days,"accuracy_pct":None,
        "label":"NO_ROWS"}
    by_anchor={}
    hits=0;counts={"DOWN":0,"SIDEWAYS":0,"UP":0};covered=0
    for row in eval_rows:
        anchor=_refit_anchor(row["signal_date"])
        if anchor not in by_anchor:
            floor=(date.fromisoformat(anchor)-timedelta(days=window_days)).isoformat()
            tallies={"DOWN":0,"SIDEWAYS":0,"UP":0}
            for past in all_train:
                if floor<=past["label_available_date"]<anchor:tallies[past["label_vol"]]+=1
            by_anchor[anchor]=max(("DOWN","SIDEWAYS","UP"),key=lambda c:(tallies[c],c)) if any(tallies.values()) else None
        prediction=by_anchor[anchor]
        counts[row["label_vol"]]+=1
        if prediction is None:continue
        covered+=1;hits+=int(prediction==row["label_vol"])
    return {"n":len(eval_rows),"covered_n":covered,"window_days":window_days,
        "accuracy_pct":100*hits/covered if covered else None,
        "class_distribution":counts,
        "label":"ROLLING_PRIOR_PLURALITY_BEFORE_EACH_REFIT_ANCHOR"}

def _probability_skill(eval_rows,probs,priors=None):
    """Brier skill against a constant-prior forecast of the SAME evaluation period.

    ``priors`` (per-row causal class priors) adds the comparison a live system
    can actually run: the period prior is an oracle that uses the block's own
    labels, the causal prior uses only labels observable before each signal.
    """
    names=("DOWN","SIDEWAYS","UP")
    if not len(eval_rows):return {"n":0,"model_brier":None,"period_prior_brier":None,"brier_skill":None}
    y=np.asarray([names.index(r["label_vol"]) for r in eval_rows],dtype=int)
    actual=np.asarray([[1. if i==j else 0. for j in range(3)] for i in y])
    p=np.asarray(probs,dtype=float)
    prior=np.bincount(y,minlength=3)/len(y)
    model_brier=float(np.mean(np.sum((p-actual)**2,axis=1)))
    prior_brier=float(np.mean(np.sum((np.broadcast_to(prior,actual.shape)-actual)**2,axis=1)))
    out={"n":len(y),"model_brier":model_brier,"period_prior_brier":prior_brier,
        "brier_skill":prior_brier-model_brier,
        "label":"positive skill means the model beats a constant class-prior forecast of the same rows"}
    if priors is not None and len(priors)==len(eval_rows):
        causal=np.asarray(priors,dtype=float)
        causal_brier=float(np.mean(np.sum((causal-actual)**2,axis=1)))
        out.update({"causal_prior_brier":causal_brier,
            "causal_prior_accuracy_pct":100*float(np.mean(causal.argmax(axis=1)==y)),
            "brier_skill_vs_causal_prior":causal_brier-model_brier,
            "causal_prior_label":"per-signal-date prior computed only from labels observable before each signal date",
            "causal_probability_edge_status":("PROBABILITY_SKILL_PRESENT_VS_CAUSAL_PRIOR"
                if model_brier<causal_brier else "NO_DEMONSTRATED_EDGE_WORSE_THAN_CAUSAL_PRIOR")})
    return out

def _backtest_model(rows,train_rows,kind,horizon,period,temperature_info=None,payoffs=None,
                    exclude_features=(),include_samples=False):
    lo,hi={"development":(SAMPLE_START,TRAIN_END),"validation":(TRAIN_END,VALID_END),"diagnostic":(VALID_END,HOLDOUT_END)}[period]
    eval_rows=[r for r in rows if r["horizon"]==horizon and lo<=r["entry_date"]<hi and r["label_available_date"]<hi]
    if not eval_rows:return {"n":0,"metrics":None,"buy_policy":None,"temperature":temperature_info}
    eval_rows.sort(key=lambda r:(r["signal_date"],r["ticker"]))
    all_train=sorted([r for r in train_rows if r["horizon"]==horizon and "label_vol" in r],key=lambda r:(r["label_available_date"],r["ticker"]))
    classes=("DOWN","SIDEWAYS","UP"); probs=np.full((len(eval_rows),3),1/3,dtype=float)
    target_probs=np.full(len(eval_rows),.5,dtype=float);valid_anchors=set()
    blocks=defaultdict(list)
    for i,r in enumerate(eval_rows):blocks[_refit_anchor(r["signal_date"])].append(i)
    for anchor in sorted(blocks):
        idx=blocks[anchor]
        past=[r for r in all_train if r["label_available_date"]<anchor and r["signal_date"]<anchor]
        if len(past)<MIN_TRAIN or len({r["label_vol"] for r in past})<3:continue
        active=tuple(k for k in _active_features(past) if k not in set(exclude_features))
        if not active:continue
        Xpast=_features(past,active);Xeval=_features([eval_rows[i] for i in idx],active)
        if kind in ("band_regression","band_winsor","band_z"):
            model=_model(kind)
            # Regression targets use the cohort's own label return (absolute net
            # return, or net return minus its basket for an excess-label cohort).
            if kind=="band_z":
                target=np.asarray([_label_net(r)/_volatility_unit(r) for r in past])
                bands=np.asarray([_band_z(eval_rows[i]) for i in idx])
            else:
                target=np.asarray([_label_net(r) for r in past])
                bands=np.asarray([_band_pct(eval_rows[i]) for i in idx])
            model.fit(Xpast,target)
            p=model.predict_proba(Xeval,bands)
        else:
            model=_model(kind);model.fit(Xpast,np.asarray([classes.index(r["label_vol"]) for r in past]))
            raw=model.predict_proba(Xeval); cls=list(model.classes_)
            p=np.zeros((len(idx),3))
            for j,c in enumerate(cls):p[:,int(c)]=raw[:,j]
        probs[idx]=p;valid_anchors.add(anchor)
        yhit=np.asarray([int(r.get("target_touch_theta_0.5")=="TARGET_FIRST") for r in past])
        if np.unique(yhit).size==2:
            hit_model=_model("multinomial_logistic");hit_model.fit(Xpast,yhit)
            hit_raw=hit_model.predict_proba(Xeval);hit_classes=list(hit_model.classes_)
            if 1 in hit_classes:target_probs[idx]=hit_raw[:,hit_classes.index(1)]
    valid=np.asarray([_refit_anchor(r["signal_date"]) in valid_anchors for r in eval_rows])
    eval_rows=[r for r,v in zip(eval_rows,valid) if v];probs=probs[valid];target_probs=target_probs[valid]
    if not eval_rows:return {"n":0,"metrics":None,"buy_policy":None,"temperature":temperature_info}
    valid=np.asarray([r["label_available_date"]<(TRAIN_END if period=="development" else VALID_END if period=="validation" else HOLDOUT_END) for r in eval_rows])
    eval_rows=[r for r,v in zip(eval_rows,valid) if v];probs=probs[valid];target_probs=target_probs[valid]
    if not eval_rows:return {"n":0,"metrics":None,"buy_policy":None,"temperature":temperature_info}
    if period=="development" and temperature_info is None:
        priors=_causal_class_priors(all_train,eval_rows)
        temperature_info=_fit_calibration(eval_rows,probs,priors)
    elif period=="development":
        priors=_causal_class_priors(all_train,eval_rows)
    else:
        priors=_causal_class_priors(all_train,eval_rows)
    temperature=float((temperature_info or {}).get("temperature",1.0))
    probs=_apply_temperature(probs,temperature)
    prior_mix=float((temperature_info or {}).get("prior_mix",0.) or 0.)
    probs=_apply_prior_mix(probs,priors,prior_mix)
    decision_tau=float((temperature_info or {}).get("decision_tau",0.) or 0.)
    balanced_predictions=_prior_corrected_decisions(probs,priors,decision_tau)
    payoffs=payoffs or (_class_payoffs(eval_rows) if period=="development" else {"DOWN":-1.,"SIDEWAYS":0.,"UP":1.})
    net_key=_metrics_net_key(eval_rows)
    theta_results={}
    for theta in THETAS:
        # Fixed +/-theta% metrics stay on the absolute net return of each row so
        # they keep their original meaning as an absolute-return reference.
        fixed=_metrics(eval_rows,probs,theta)
        theta_results[str(theta)]={"metrics":fixed}
    vol_metrics=_metrics(eval_rows,probs,label_key="label_vol",net_key=net_key)
    target_actual=np.asarray([int(r.get("target_touch_theta_0.5")=="TARGET_FIRST") for r in eval_rows],dtype=float)
    target_report={"n":len(eval_rows),"observed_target_first_pct":100*float(target_actual.mean()),
      "mean_predicted_probability_pct":100*float(target_probs.mean()),"brier":float(np.mean((target_probs-target_actual)**2))}
    vol_metrics["target_hit_probability"]=target_report
    vol_metrics["causal_prior_baseline"]=_causal_prior_baseline(all_train,eval_rows)
    vol_metrics["probability_skill"]=_probability_skill(eval_rows,probs,priors)
    vol_metrics["class_balanced_decision"]=_balanced_decision_report(eval_rows,balanced_predictions,priors,
        decision_tau)
    vol_metrics["plain_argmax_decision"]=_balanced_decision_report(eval_rows,probs.argmax(axis=1),priors,0.)
    policies=_policy_table(eval_rows,probs,payoffs)
    result={"n":len(eval_rows),"metrics":vol_metrics,"fixed_band_metrics":{k:v["metrics"] for k,v in theta_results.items()},
      "theta_results":theta_results,"buy_policy_candidates":policies,"buy_policy":None,
      "temperature":temperature_info,"class_payoffs_from_development":payoffs,
      "policy_selection_scope":"development only; thresholds are spread, sideways-probability, and net EV after existing costs",
      "probability_sample":{"p_up_p10_p50_p90":np.percentile(probs[:,2],[10,50,90]).tolist(),
        "p_down_p10_p50_p90":np.percentile(probs[:,0],[10,50,90]).tolist(),
        "p_sideways_p10_p50_p90":np.percentile(probs[:,1],[10,50,90]).tolist()}}
    if include_samples:
        result["evaluation_samples"]=[{"label":row["label_vol"],
            "prediction":classes[int(balanced)],
            "argmax_prediction":classes[int(np.argmax(prob))],
            "probabilities":{"DOWN":float(prob[0]),"SIDEWAYS":float(prob[1]),"UP":float(prob[2])},
            "causal_prior":{"DOWN":float(prior[0]),"SIDEWAYS":float(prior[1]),"UP":float(prior[2])},
            "earnings_momentum":row.get("earnings_momentum")}
            for row,prob,prior,balanced in zip(eval_rows,probs,priors,balanced_predictions)]
    return result

def _balanced_decision_report(rows,predictions,priors,tau):
    """Argmax versus class-balanced decision on the SAME calibrated probabilities.

    The plain argmax of a calibrated model can stay pinned to one class when its
    class prior is skewed. The balanced rule divides by the causal prior so a
    class the prior already expects has to clear a higher probability bar; it is
    reported next to the argmax instead of replacing it, so the accuracy cost of
    un-skewing the predictions stays visible.
    """
    names=("DOWN","SIDEWAYS","UP")
    if not len(rows or []):
        return {"n":0}
    y=np.asarray([names.index(r["label_vol"]) for r in rows],dtype=int)
    predictions=np.asarray(predictions,dtype=int)
    out={"n":len(y),"tau":float(tau),"rule":"argmax(p / causal_prior**tau)",
         "predicted_counts":{name:int(np.sum(predictions==j)) for j,name in enumerate(names)},
         "accuracy_pct":100*float(np.mean(predictions==y)),
         "class_metrics":{}}
    for j,name in enumerate(names):
        correct=int(np.sum((predictions==j)&(y==j)))
        predicted=int(np.sum(predictions==j));actual=int(np.sum(y==j))
        out["class_metrics"][name]={"predicted":predicted,"actual":actual,"correct":correct,
            "precision_pct":100*correct/predicted if predicted else 0.,
            "recall_pct":100*correct/actual if actual else 0.}
    return out

def _ablation_period_summary(result,policy_key):
    metrics=result.get("metrics") or {}
    policies=result.get("buy_policy_candidates") or {}
    policy=policies.get(policy_key) if policy_key else None
    return {"n":result.get("n",0),"accuracy_pct":metrics.get("accuracy_pct"),
      "majority_class_baseline_accuracy_pct":metrics.get("majority_class_baseline_accuracy_pct"),
      "multiclass_log_loss":metrics.get("multiclass_log_loss"),
      "constant_prior_entropy_nats":metrics.get("constant_prior_entropy_nats"),
      "class_brier":metrics.get("class_brier"),"class_ece":metrics.get("class_ece"),
      "selected_policy":policy_key,"policy_metrics":policy}

def _training_rows_for(cohort,pool,pool63):
    """Use same-cohort ordinary US equities to train the SOXL audit model."""
    return pool if cohort=="US" else pool63

def _comparison_fit_spec(cohort,fit_name,pool,pool63,eval_rows):
    """Resolve training rows per universe comparison while freezing evaluation rows."""
    if cohort=="US":
        training=pool
        scope="available full ordinary-US equity cohort; SOXL evaluation-only (same training pool for both fit labels)"
    elif fit_name=="63_audit_symbols":
        training=pool63
        scope="frozen 63 audit benchmark symbols within this Korean cohort"
    elif fit_name=="available_full_universe":
        training=pool
        scope="all available ordinary shares in this Korean cohort"
    else:
        raise ValueError(f"unknown universe fit: {fit_name}")
    return {"training_rows":training,"evaluation_rows":eval_rows,
        "training_rows_n":len(training),"training_scope":scope,
        "evaluation_rows_n":len(eval_rows),
        "evaluation_tickers_n":len({str(row["ticker"]) for row in eval_rows})}

def _flow_ablation_metadata(model,horizon,policy_key,fit_spec):
    return {"same_model":model,"same_horizon":horizon,"policy_key":policy_key,
        "policy_threshold_source":"frozen 63-symbol development selection; unchanged",
        "training_rows_n":fit_spec["training_rows_n"],
        "training_scope":fit_spec["training_scope"],
        "evaluation_rows_n":fit_spec["evaluation_rows_n"],
        "evaluation_tickers_n":fit_spec["evaluation_tickers_n"]}

def _selection_rows_for(cohort,pool,benchmark_rows):
    """Select US model/horizon on full-universe OOS-development forecasts."""
    return pool if cohort=="US" else benchmark_rows

def run_expanded_direction_audit(*,flow_frames=None,flow_source_coverage=None,macro_frames=None,
                                macro_diagnostics=None,earnings_by_ticker=None,
                                earnings_key_present=False):
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from opendart import earnings_features_for_signal
    universe=_ticker_set(); audit=_audit_symbols(universe); rows=[]; loaded=[]; failures=[]; special_eval={}
    flow_frames=flow_frames or {};flow_source_coverage=flow_source_coverage or {}
    earnings_by_ticker=earnings_by_ticker or {}
    sources=universe.pop("universe_sources",{}); universe_errors=universe.pop("universe_errors",{})
    jobs=[(c,item) for c,items in universe.items() for item in items]
    frames_by_cohort={c:{} for c in universe}
    with ThreadPoolExecutor(max_workers=12) as pool:
      futs={pool.submit(get_historical_prices,item[0],item[2],start_date=SAMPLE_START,min_bars=250):(cohort,item)
            for cohort,item in jobs}
      for fut in as_completed(futs):
        cohort,(ticker,name,market,sector)=futs[fut]
        try:
            frame,source=fut.result()
            first=pd.Timestamp(frame.index.min()).date().isoformat()
            if pd.Timestamp(frame.index.max()).date().isoformat()<HOLDOUT_END:raise ValueError("history ends before fixed HOLDOUT_END")
            frames_by_cohort[cohort][ticker]=frame
            made=_make_rows(frame,ticker,market,cohort);rows.extend(made)
            loaded.append({"ticker":ticker,"cohort":cohort,"first_date":first,"source":source,"rows":len(made)})
        except Exception as exc:failures.append({"ticker":ticker,"cohort":cohort,"reason":str(exc)[:220]})
    audit_tickers=set().union(*audit.values())
    # Keep the mandated leveraged ETF in the 63-instrument evaluation. It is
    # never mixed into the ordinary-US cross-section used to train the model.
    for ticker,name,market in _mandatory_market_items(MANDATORY_US,"US"):
        try:
            frame,source=get_historical_prices(ticker,market,start_date=SAMPLE_START,min_bars=250)
            frames_by_cohort.setdefault("US",{})[ticker]=frame
            special_eval.setdefault("US",[]).extend(_make_rows(frame,ticker,market,"US"))
            loaded.append({"ticker":ticker,"cohort":"US_eval_only","first_date":str(frame.index.min().date()),"source":source,"rows":len(special_eval["US"])})
        except Exception as exc:failures.append({"ticker":ticker,"cohort":"US_eval_only","reason":str(exc)[:220]})
    for ticker,name,market in _mandatory_market_items(MANDATORY_KR,"KR"):
        try:
            frame,source=get_historical_prices(ticker,market,start_date=SAMPLE_START,min_bars=250)
            frames_by_cohort.setdefault("TIGER_ETF",{})[ticker]=frame
            special_eval.setdefault("TIGER_ETF",[]).extend(_make_rows(frame,ticker,market,"TIGER_ETF"))
            loaded.append({"ticker":ticker,"cohort":"TIGER_ETF_eval_only","first_date":str(frame.index.min().date()),"source":source,"rows":len(special_eval["TIGER_ETF"])})
        except Exception as exc:failures.append({"ticker":ticker,"cohort":"TIGER_ETF_eval_only","reason":str(exc)[:220]})
    # Same-date US cross-sectional ranks include SOXL as a benchmark member;
    # ETF-only ranks remain within the frozen two-ETF cohort.
    us_symbols={r["ticker"] for r in special_eval.get("US",[])}
    rows=_rank_features(rows+special_eval.get("US",[]))
    special_eval["US"]=[r for r in rows if r["ticker"] in us_symbols and r["cohort"]=="US"]
    rows=[r for r in rows if r["ticker"] not in us_symbols]
    bycohort={c:[r for r in rows if r["cohort"]==c] for c in universe}
    special_eval["TIGER_ETF"]=_rank_features(special_eval.get("TIGER_ETF",[]))
    if macro_frames is None:
        macro_frames,macro_diagnostics=get_market_series_bundle(start_date=SAMPLE_START,end_date=HOLDOUT_END)
    extended_pool_collection=_collect_extended_pool_flows(universe,flow_frames,
        flow_source_coverage=flow_source_coverage)
    _attach_context_features(rows,frames_by_cohort,macro_frames,flow_frames)
    for c in ("US","TIGER_ETF"):
        _attach_context_features(special_eval.get(c,[]),{c:frames_by_cohort.get(c,{})},macro_frames,flow_frames)
    # Direction labels: KOSPI keeps its absolute net-return label; KOSDAQ is
    # labelled on the excess return over its own same-date basket. Attaching it
    # before the band is frozen keeps one band per cohort on one label return.
    label_mode_report=_attach_label_returns(rows)
    volatility_k=_freeze_volatility_labels(rows+special_eval.get("US",[])+special_eval.get("TIGER_ETF",[]))
    # Rows a thin basket left unlabelled take no part in fitting or evaluation.
    rows=[r for r in rows if r.get("label_vol") in ("DOWN","SIDEWAYS","UP")]
    bycohort={c:[r for r in rows if r["cohort"]==c] for c in universe}
    equity_loaded=sum(1 for row in loaded if row.get("cohort") in universe)
    out={"universe_expansion":{"requested":sum(map(len,universe.values())),"loaded":equity_loaded,
      "evaluation_only_loaded":len(loaded)-equity_loaded,
      "failures":failures,"loaded_tickers":loaded,"training_pool_candidate_rows":{c:len(v) for c,v in bycohort.items()},
      "universe_sources":sources,"universe_lookup_errors":universe_errors,
      "audit63_scored_symbols":{c:len((set(r["ticker"] for r in v)|
          (set(r["ticker"] for r in special_eval.get("US",[])) if c=="US" else set()))&audit.get(c,set()))
          for c,v in bycohort.items()},
      "frozen_audit_benchmark":{c:list(v) for c,v in AUDIT_63_BENCHMARK.items()},
      "benchmark_missing_from_training_pool":{c:sorted(set(AUDIT_63_BENCHMARK.get({"US":"SOXL"}.get(c,c),()))-
          set(x["ticker"] for x in bycohort.get(c,[]))) for c in bycohort},
      "benchmark_evaluation_coverage":{"US":{"benchmark_key":"SOXL","evaluation_only_tickers":sorted({r["ticker"] for r in special_eval.get("US",[])}),
          "training_pool_exclusion_reason":"SOXL is retained as the 63-symbol evaluation instrument and excluded from the ordinary-US training pool."}},
      "flow_collection_coverage":flow_source_coverage,
      "macro_series_diagnostics":macro_diagnostics or {},
      "feature_point_in_time":"All row features use signal-day close or earlier. Direction3 macro features use strict prior US session closes for VIX, USD/KRW, S&P 500, SOX; cohort-local Korean index features use the signal-day Korean close.",
      "cross_sectional_scope":"same signal date and same cohort only; missing symbols are excluded from that date rank",
      "periods":{"development_end":TRAIN_END,"validation_end":VALID_END,"diagnostic_end":HOLDOUT_END}},
      "label_variants":{"entry":"t+1 open","exit":"t+h close","horizons":list(HORIZONS),
      "cost_pct":COST_PCT,"legacy_fixed_thresholds_pct":list(THETAS),"sensitivity_selected_horizon":None,
      "label_mode_by_cohort":{c:_label_mode(c) for c in bycohort},
      "excess_return_label":{c:label_mode_report[c] for c in label_mode_report
        if label_mode_report[c].get("mode")=="excess_vs_cohort_mean"},
      "absolute_return_cohorts":[c for c in bycohort if _label_mode(c)=="absolute_return"],
      "volatility_band":{"formula":"k_cohort * signal_day_realized_vol20_pct * sqrt(h/20)","target_development_sideways_share_pct":100*SIDEWAYS_TARGET_SHARE,
        "cohort_parameters":volatility_k}},
      "direction3":{"label_definition":"net return after existing cost on the cohort's label return (KOSPI: the stock's own net return; KOSDAQ: net return minus the same signal-date/horizon equal-weighted cohort basket net return); volatility-scaled symmetric SIDEWAYS band; cohort k fitted using only outcomes available before TRAIN_END","cohort_k":volatility_k,"probability_calibration":{},"policy_results":{},
        "minimum_development_ic_for_edge_label":MIN_DEVELOPMENT_IC},
      "extended_pool_collection":extended_pool_collection,
      "minimum_development_ic_for_edge_label":MIN_DEVELOPMENT_IC,
      "selection_rule":"Development-only walk-forward: select one model/horizon by mean datewise Spearman(P(UP), realized net return), requiring positive IC, at least 100 top-15% development candidates, and development three-way accuracy at least the live-computable causal class prior. Model grid covers multinomial logistic, random forest, band regression (predicted net return mapped through the frozen volatility band), a winsorised/MAD variant and a volatility-normalised (z-return) variant. Calibrate multiclass probabilities on development OOF: temperature, shrinkage toward the per-signal-date causal class prior (weight fitted on development Brier), and a class-balanced decision threshold fitted on development accuracy. Select spread + sideways-probability + net expected-value policy on development only; all selected policy OOS results are reported.",
      "feature_coverage":{"feature_names":list(FEATURES),"fixed_centers":CENTERS,"fixed_scales":SCALES,"feature_count":len(FEATURES),"cross_section_count_key":"xs_count","constant_column_policy":"recomputed from each past-only training snapshot and excluded when std is zero"}}
    direction3_eval_samples_by_cohort={}
    for cohort,pool in bycohort.items():
        eval_rows=[r for r in pool if r["ticker"] in audit.get(cohort,set())]
        eval_rows.extend(special_eval.get(cohort,[]))
        for row in eval_rows:
            if cohort in ("KOSPI","KOSDAQ"):
                cached=earnings_by_ticker.get(str(row["ticker"]).zfill(6),{})
                diagnostics=cached.get("diagnostics",{})
                earnings=earnings_features_for_signal(cached.get("records",[]),row["signal_date"],
                    corp_code=diagnostics.get("corp_code"),abstain=not earnings_key_present)
            else:
                earnings=earnings_features_for_signal([],row["signal_date"],corp_code=None,
                    not_applicable=True)
            # Diagnostic grouping only: these values are not in FEATURES and do
            # not participate in direction3 fitting, selection, or thresholds.
            row["earnings_momentum"]=earnings.get("earnings_momentum")
        pool63=[r for r in eval_rows]
        # SOXL is evaluation-only; selection and every forecast for it must be
        # trained from the available ordinary-US cohort, never the 162-row ETF.
        selection_training=_training_rows_for(cohort,pool,pool63)
        selection_rows=_selection_rows_for(cohort,pool,eval_rows)
        choices=[];development_cache={}
        for horizon in HORIZONS:
          for model in MODEL_KINDS:
            # Select the US model/horizon on walk-forward development forecasts
            # across the same-cohort ordinary-stock pool. SOXL is then a truly
            # separate one-symbol evaluation set, not the model-selection set.
            dev=_backtest_model(selection_rows,selection_training,model,horizon,"development")
            development_cache[(model,horizon)]=dev
            met=dev.get("metrics") or {}; ic=met.get("datewise_spearman_mean"); top=met.get("top15_mean_net_pct")
            score=(float(ic)+.01*float(top)) if ic is not None and top is not None and met.get("top15_n",0)>=100 else -1e9
            choices.append((score,model,horizon))
        if cohort=="US":
            # Diagnostics must still score evaluation-only SOXL from the full
            # ordinary-US training pool even when no development edge exists.
            # This selects a diagnostic model only; the independent no-edge
            # gate below still prevents a buy policy from being selected.
            eligible_choices=[x for x in choices if x[0]>-1e8 and (development_cache[(x[1],x[2])].get("n",0)>0)]
        else:
            # Selection stays development-only; the added requirement is that the
            # calibrated model must also beat the live-computable causal prior on
            # development three-way accuracy, so a variant that only wins the
            # ranking proxy cannot displace a model that classifies better.
            eligible_choices=[x for x in choices if x[0]>0 and _beats_causal_prior_on_development(
                development_cache[(x[1],x[2])])]
        diagnostic_fallback=False
        if not eligible_choices and EXCESS_LABEL_DIAGNOSTIC_FALLBACK and _label_mode(cohort)=="excess_vs_cohort_mean":
            # Report the exploration label even when nothing clears the rule. The
            # status below says so explicitly and no edge is claimed.
            eligible_choices=[x for x in choices if x[0]>-1e8 and development_cache[(x[1],x[2])].get("n",0)>0]
            diagnostic_fallback=True
        chosen=max(eligible_choices,default=(-1e9,None,None))
        chosen_ic=((development_cache[(chosen[1],chosen[2])].get("metrics") or {}).get("datewise_spearman_mean")
                   if chosen[1] is not None else None)
        cohort_report={"selected_model_development_only":chosen[1],"selected_horizon_development_only":chosen[2],
          "label_mode":_label_mode(cohort),
          "label_return_key":("net_label_pct" if _label_mode(cohort)=="excess_vs_cohort_mean" else "net_pct"),
          "label_mode_detail":label_mode_report.get(cohort),
          "training_pool_scope":"available full ordinary-US equity cohort; SOXL is evaluation-only" if cohort=="US" else "63 audit cohort for selection; full cohort reported as separate training comparison",
          "selected_development_datewise_spearman":chosen_ic,
          "model_selection_status":(_edge_label_status(chosen[1] is not None,chosen_ic)
            if not diagnostic_fallback else EXCESS_LABEL_FALLBACK_STATUS),
          "selected_by_full_development_rule":not diagnostic_fallback,
          "minimum_development_ic_for_edge_label":MIN_DEVELOPMENT_IC,
          "selected_model_horizon_development_score":chosen[0],"development_model_grid":{
            f"{model}_h{h}":{"datewise_spearman_mean":(v.get("metrics") or {}).get("datewise_spearman_mean"),
              "datewise_spearman_n_days":(v.get("metrics") or {}).get("datewise_spearman_n_days"),
              "accuracy_pct":(v.get("metrics") or {}).get("accuracy_pct"),
              "majority_class_baseline_accuracy_pct":(v.get("metrics") or {}).get("majority_class_baseline_accuracy_pct"),
              "multiclass_log_loss":(v.get("metrics") or {}).get("multiclass_log_loss"),
              "constant_prior_entropy_nats":(v.get("metrics") or {}).get("constant_prior_entropy_nats"),
              "score":s}
            for s,model,h in choices for v in [development_cache[(model,h)]]},
          "universe_comparison":{},"periods":{}}
        selected_policy=None;selection_dev=None;selection_payoffs=None;selection_grid={}
        if chosen[1] is not None:
            h=chosen[2]
            selection_dev=development_cache[(chosen[1],h)]
            selection_payoffs=selection_dev.get("class_payoffs_from_development") or _class_payoffs(
                [r for r in selection_training if r["horizon"]==h and r["label_available_date"]<TRAIN_END])
            selection_grid=selection_dev.get("buy_policy_candidates",{})
            eligible=[(v.get("mean_net_pct"),key) for key,v in selection_grid.items()
                      if v.get("n",0)>=MIN_TRAIN and v.get("mean_net_pct") is not None and v.get("mean_net_pct")>0]
            selected_policy=max(eligible,default=(None,None),key=lambda x:(x[0],x[1]))[1]
            cohort_report["development_policy_grid"]=selection_grid
            cohort_report["selected_policy_thresholds"]={"policy":selected_policy,
              "calibration":selection_dev.get("temperature"),"class_payoffs":selection_payoffs,
              "diagnostic_excess_label_fallback":bool(diagnostic_fallback),
              "selection_rule":"selected from the frozen 63-symbol development evaluation only; reused unchanged across training-universe and flow ablations"}
        cohort_report["fit_metadata_by_universe"]={}
        for fit_name in ("63_audit_symbols","available_full_universe"):
          spec=_comparison_fit_spec(cohort,fit_name,pool,pool63,eval_rows)
          training=spec["training_rows"];fixed_eval_rows=spec["evaluation_rows"]
          fit_meta={k:v for k,v in spec.items() if k not in ("training_rows","evaluation_rows")}
          cohort_report["fit_metadata_by_universe"][fit_name]=fit_meta
          comparisons={}
          audit_eval_samples={period:[] for period in ("development","validation","diagnostic")}
          if chosen[1] is not None:
            h=chosen[2]
            if fit_name=="63_audit_symbols" or cohort=="US":
                fit_dev=selection_dev
            else:
                fit_dev=_backtest_model(fixed_eval_rows,training,chosen[1],h,"development")
            fit_payoffs=fit_dev.get("class_payoffs_from_development") or _class_payoffs(
                [r for r in training if r["horizon"]==h and r["label_available_date"]<TRAIN_END])
            dev=_backtest_model(fixed_eval_rows,training,chosen[1],h,"development",
                temperature_info=fit_dev.get("temperature"),payoffs=fit_payoffs,
                include_samples=fit_name=="63_audit_symbols")
            periods={"development":dev}
            for period in ("validation","diagnostic"):
                periods[period]=_backtest_model(fixed_eval_rows,training,chosen[1],h,period,
                  temperature_info=fit_dev.get("temperature"),payoffs=fit_payoffs,
                  include_samples=fit_name=="63_audit_symbols")
            if fit_name=="63_audit_symbols":
                audit_eval_samples={period:raw.get("evaluation_samples",[])
                    for period,raw in periods.items()}
            for period,raw in periods.items():
                met=raw.get("metrics");grid=raw.get("buy_policy_candidates",{})
                comparisons[period]={"n":raw.get("n",0),"metrics":met,"fixed_band_metrics":raw.get("fixed_band_metrics"),
                  "selected_policy":selected_policy,"selected_policy_metrics":grid.get(selected_policy) if selected_policy else None,
                  "training_rows_n":len(training),"training_scope":spec["training_scope"],
                  "evaluation_rows_n":len(fixed_eval_rows),"evaluation_tickers_n":spec["evaluation_tickers_n"],
                  "all_development_grid_policy_results":grid if period=="development" else None,
                  "probability_sample":raw.get("probability_sample")}
            if fit_name=="63_audit_symbols":
                cohort_report["selected_policy_thresholds"].update(
                    {"calibration":fit_dev.get("temperature"),"class_payoffs":fit_payoffs})
            else:
                cohort_report["full_universe_development_policy_grid"]=fit_dev.get("buy_policy_candidates",{})
            cohort_report.setdefault("calibration_by_universe",{})[fit_name]=fit_dev.get("temperature")
            cohort_report.setdefault("policy_selection_status_by_universe",{})[fit_name]=(
                "DEVELOPMENT_POLICY_SELECTED_FROM_63_FIXED_FOR_COMPARISON" if selected_policy else "NO_DEVELOPMENT_EDGE")
          else:
            comparisons={p:{"n":0,"metrics":None,"selected_policy":None,"selected_policy_metrics":None,
                "training_rows_n":len(training),"training_scope":spec["training_scope"],
                "evaluation_rows_n":len(fixed_eval_rows),"evaluation_tickers_n":spec["evaluation_tickers_n"]}
                for p in ("development","validation","diagnostic")}
            cohort_report.setdefault("policy_selection_status_by_universe",{})[fit_name]="NO_DEVELOPMENT_EDGE"
          cohort_report["universe_comparison"][fit_name]=comparisons
          if fit_name=="63_audit_symbols":
              direction3_eval_samples_by_cohort[cohort]=audit_eval_samples
        # Feature ablation reuses the selected model, horizon, label, and
        # development-selected policy threshold. Only flow inputs are removed;
        # validation/diagnostic are reported, never used for reselection.
        if chosen[1] is not None:
            h=chosen[2];base_dev=development_cache[(chosen[1],h)]
            policy_key=(cohort_report.get("selected_policy_thresholds") or {}).get("policy")
            audit_spec=_comparison_fit_spec(cohort,"63_audit_symbols",pool,pool63,eval_rows)
            ablation={**_flow_ablation_metadata(chosen[1],h,policy_key,audit_spec),
              "excluded_features":list(FLOW_FEATURES),"periods":{}}
            noflow_selection_dev=_backtest_model(selection_rows,selection_training,chosen[1],h,"development",exclude_features=FLOW_FEATURES)
            noflow_temp=noflow_selection_dev.get("temperature");noflow_payoffs=noflow_selection_dev.get("class_payoffs_from_development")
            audit_dev_with=(base_dev if cohort!="US" else _backtest_model(eval_rows,selection_training,chosen[1],h,"development",
                temperature_info=base_dev.get("temperature"),payoffs=base_dev.get("class_payoffs_from_development")))
            audit_dev_without=(noflow_selection_dev if cohort!="US" else _backtest_model(eval_rows,selection_training,chosen[1],h,"development",
                temperature_info=noflow_temp,payoffs=noflow_payoffs,exclude_features=FLOW_FEATURES))
            ablation["periods"]["development"]={"with_flow":_ablation_period_summary(audit_dev_with,policy_key),
                "without_flow":_ablation_period_summary(audit_dev_without,policy_key)}
            for period in ("validation","diagnostic"):
                train_for=_training_rows_for(cohort,pool,pool63)
                raw_with=(development_cache[(chosen[1],h)] if period=="development" else
                    _backtest_model(eval_rows,train_for,chosen[1],h,period,
                      temperature_info=base_dev.get("temperature"),payoffs=base_dev.get("class_payoffs_from_development")))
                raw_without=_backtest_model(eval_rows,train_for,chosen[1],h,period,
                    temperature_info=noflow_temp,payoffs=noflow_payoffs,exclude_features=FLOW_FEATURES)
                ablation["periods"][period]={"with_flow":_ablation_period_summary(raw_with,policy_key),
                    "without_flow":_ablation_period_summary(raw_without,policy_key)}
            full_spec=_comparison_fit_spec(cohort,"available_full_universe",pool,pool63,eval_rows)
            full_training=full_spec["training_rows"];full_eval=full_spec["evaluation_rows"]
            full_dev_with=_backtest_model(full_eval,full_training,chosen[1],h,"development")
            full_dev_without=_backtest_model(full_eval,full_training,chosen[1],h,"development",
                exclude_features=FLOW_FEATURES)
            full_with_temp=full_dev_with.get("temperature");full_with_payoffs=full_dev_with.get("class_payoffs_from_development")
            full_without_temp=full_dev_without.get("temperature");full_without_payoffs=full_dev_without.get("class_payoffs_from_development")
            full_periods={"development":{"with_flow":_ablation_period_summary(full_dev_with,policy_key),
                "without_flow":_ablation_period_summary(full_dev_without,policy_key)}}
            for period in ("validation","diagnostic"):
                full_with=_backtest_model(full_eval,full_training,chosen[1],h,period,
                    temperature_info=full_with_temp,payoffs=full_with_payoffs)
                full_without=_backtest_model(full_eval,full_training,chosen[1],h,period,
                    temperature_info=full_without_temp,payoffs=full_without_payoffs,
                    exclude_features=FLOW_FEATURES)
                full_periods[period]={"with_flow":_ablation_period_summary(full_with,policy_key),
                    "without_flow":_ablation_period_summary(full_without,policy_key)}
            ablation["full_universe_training"]={**_flow_ablation_metadata(chosen[1],h,policy_key,full_spec),
              "periods":full_periods}
            cohort_report["flow_ablation"]=ablation
        else:
            cohort_report["flow_ablation"]={"status":"NO_DEVELOPMENT_MODEL_SELECTED","same_model_comparison":False,
                "full_universe_training":{"status":"NO_DEVELOPMENT_MODEL_SELECTED","same_model_comparison":False}}
        if diagnostic_fallback:
            cohort_report["policy_status"]="DIAGNOSTIC_EXCESS_LABEL_EXPLORATION_NO_POLICY_CLAIM"
        else:
            cohort_report["policy_status"]=("SELECTED_ON_DEVELOPMENT_ONLY" if any(v for d in cohort_report.get("selected_buy_cut_by_development",{}).values() for v in d.values())
          else ("DEVELOPMENT_POLICY_SELECTED" if chosen[1] is not None and cohort_report.get("selected_policy_thresholds",{}).get("policy") else "NO_DEVELOPMENT_EDGE"))
        cohort_report["selected_model_policy_validation_diagnostic"]={name:cohort_report["universe_comparison"].get(name,{}) for name in ("63_audit_symbols","available_full_universe")}
        cover={}
        dev_rows=[r for r in pool if SAMPLE_START<=r["signal_date"] and r["label_available_date"]<TRAIN_END]
        excluded=[]
        for key in FEATURES:
            vals=np.asarray([float(r.get(key,np.nan)) for r in dev_rows],dtype=float)
            finite=vals[np.isfinite(vals)]
            cover[key]={"rows_n":len(vals),"nonmissing_pct":100*len(finite)/len(vals) if len(vals) else 0,
              "unique_values_n":int(np.unique(finite).size),"standard_deviation":float(np.std(finite)) if len(finite) else None,
              "auto_excluded_constant":bool(len(finite)==0 or np.std(finite)<=1e-12)}
            if len(finite)==0 or np.std(finite)<=1e-12:excluded.append(key)
        cover["excluded_constant_columns"]=excluded
        for key in excluded:print(f"DIRECTION_FEATURE_AUTO_EXCLUDED cohort={cohort} feature={key} reason=zero_development_std",flush=True)
        cohort_report["feature_coverage"]=cover
        out["direction3"][cohort]=cohort_report
    out["direction3"]["TIGER_ETF"]={"status":"NO_INDEPENDENT_TRAINING_UNIVERSE","evaluation_rows_n":len(special_eval.get("TIGER_ETF",[])),
      "reason":"ETFs are kept as a separate cohort; the expanded 300-equity pool does not mix them with Korean ordinary shares."}
    out["direction3"]["calibration_scope"]=("development OOF probabilities only: one scalar temperature, one causal-prior shrinkage weight "
        f"(prior window {PRIOR_SHRINKAGE_WINDOW_DAYS} days from labels observable before each signal date) and one class-balanced "
        "decision threshold per selected model/horizon/training universe; applied unchanged to validation and diagnostic")
    out["direction3"]["probability_calibration"]={cohort:value.get("calibration_by_universe",{})
        for cohort,value in out["direction3"].items() if isinstance(value,dict) and "universe_comparison" in value}
    out["direction3"]["policy_results"]={cohort:{universe_name:periods for universe_name,periods in value.get("universe_comparison",{}).items()}
        for cohort,value in out["direction3"].items() if isinstance(value,dict) and "universe_comparison" in value}
    out["direction3"]["label_definition_comparison"]={cohort:{"label_mode":_label_mode(cohort),"volatility_band":v,
      "label_return":("net_pct - same signal-date/horizon equal-weight cohort basket mean net_pct"
        if _label_mode(cohort)=="excess_vs_cohort_mean" else "net_pct"),
      "label_mode_detail":label_mode_report.get(cohort),
      "fixed_pct_band_development_class_counts":{
        str(theta):{name:sum(r.get(f"label_{theta}")==name for r in bycohort.get(cohort,[]) if r["label_available_date"]<TRAIN_END)
                    for name in ("DOWN","SIDEWAYS","UP")} for theta in THETAS}}
      for cohort,v in volatility_k.items()}
    out["direction3"]["_evaluation_samples"]=direction3_eval_samples_by_cohort
    # Preserve readable, auditable source/data failures separately.
    Path("data").mkdir(exist_ok=True)
    return out

if __name__=="__main__":
    result=run_expanded_direction_audit()
    Path("data/expanded-direction-latest.json").write_text(json.dumps(result,ensure_ascii=False,indent=2,default=str),encoding="utf-8")
    print(json.dumps({"requested":result["universe_expansion"]["requested"],"loaded":result["universe_expansion"]["loaded"],"failures":len(result["universe_expansion"]["failures"])},ensure_ascii=False),flush=True)
