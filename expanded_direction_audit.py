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
DECISION_TAU_GRID=(0.,.05,.1,.15,.2,.25,.3,.35,.4,.45,.5,.55,.6,.65,.7,.75,.8,.85,.9,.95,
                   1.,1.05,1.1,1.15,1.2,1.3,1.4,1.5)
# Temperature is fitted in log space. The old (-1.5, 1.5) box capped T at e^1.5
# = 4.48 and every cohort in the PR #18 audit saturated there, so the shipped
# temperature was a boundary value rather than a fitted one.
TEMPERATURE_LOG_BOUNDS=(-2.,4.)
# The probability-level prior mixture is reported as a reference but no longer
# feeds the decision: mixing toward the per-date prior at weight ~0.7 compressed
# p(UP)-p(DOWN) to +-0.03, which is what pinned every row to one class.
MAX_REFERENCE_PRIOR_MIX=.5
# Candidate causal-prior memories (exponential half-life in days) for the class
# prior used by the decision rule. Chosen on development log-loss only.
PRIOR_HALF_LIFE_GRID=(40.,80.,160.,250.,400.)
# A decision whose single most predicted class exceeds these shares is reported
# as pinned/dominant instead of being passed off as a calibrated classifier.
DECISION_DOMINANT_SHARE=.90
DECISION_PINNED_SHARE=.99
SIDEWAYS_TARGET_SHARE=.30
# Causal regime/trend gate on the direction3 decision. After the calibration and
# decision stages were separated the raw model still put p(UP)-p(DOWN) at
# +0.57..+0.93 on the 2026 diagnostics, so no monotone correction can move the
# decision and the model calls UP for almost every instrument while the market
# trend has turned down. A separate trend rule may therefore move an UP decision
# to the conservative class when the signal day's own index state is risk-off.
# Every threshold is fitted on development OOF decisions only and then applied
# unchanged to validation and diagnostic, so this is a risk rule, not a leak.
#
# Risk-off means the short-term trend is weak and the longer trend is broken.
# Three pre-declared states are available, and the development fit chooses one:
#   index                 cohort index: 20-day return weak AND (below its 200-day
#                         mean OR more than a set drawdown below its 120-day high)
#   index_and_instrument  the same index state AND the instrument itself below
#                         its own 200-day mean
#   instrument            the instrument's own 20-day return weak AND it is below
#                         its own 200-day mean
# The drawdown clause exists because an index can sit above its 200-day mean
# while the stocks underneath it are already falling.
REGIME_GATE_STATES=("index","index_and_instrument","instrument")
REGIME_GATE_MA200_GAP_GRID=(-4.,-2.,-1.,0.,2.)
REGIME_GATE_DRAWDOWN_GRID=(-4.,-8.,-12.,-16.)
REGIME_GATE_INDEX_RET20_GRID=(2.,0.,-1.,-2.,-4.)
REGIME_GATE_TARGETS=("DOWN","SIDEWAYS","SECOND_CHOICE")
# A gate that fires on only a handful of development rows is a coincidence, not
# a rule; and a gate that pins the whole development decision to one class trades
# the UP pin for a DOWN pin, which the F1 guard below rejects.
REGIME_GATE_MIN_DEV_OVERRIDES=200
REGIME_GATE_MIN_ACCURACY_GAIN_PP=.2
REGIME_GATE_MAX_MACRO_F1_DROP_PP=.5
REGIME_GATE_FEATURES=("market_index_ma200_gap_pct","market_drawdown_120_pct",
    "market_index_return_20d_pct")
MIN_DEVELOPMENT_IC=.01
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

def _development_accuracy(development_result):
    """Three-way accuracy of the development-calibrated decision rule."""
    metrics=(development_result or {}).get("metrics") or {}
    decision=metrics.get("decision") or metrics.get("class_balanced_decision") or {}
    value=decision.get("accuracy_pct")
    return float(value) if value is not None else None

def _beats_causal_prior_on_development(development_result):
    """Model must beat the causal prior on development accuracy AND on IC sign.

    The decision also has to be non-degenerate: a rule that names one class for
    every development row is not a classifier, even when that class happens to
    carry the development base rate.
    """
    metrics=(development_result or {}).get("metrics") or {}
    if not (metrics.get("datewise_spearman_mean") or 0)>0:
        return False
    decision=metrics.get("decision") or metrics.get("class_balanced_decision") or {}
    accuracy=decision.get("accuracy_pct")
    diversity=decision.get("predicted_share_pct") or {}
    if len([v for v in diversity.values() if v>=1.])<2:
        return False
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
        residual=self._residual_scale(X,y)
        if self.scale=="mad":
            mad=float(np.median(np.abs(residual-np.median(residual))))*1.4826
            self.sigma=float(max(mad,np.std(residual)*.25,1e-6))
        else:
            self.sigma=float(max(np.std(residual),1e-6))
        return self
    def _residual_scale(self,X,y,folds=5):
        """Out-of-fold residual vector, so sigma is a forecast error.

        In-sample residuals shrink with the number of fitted coefficients and
        make P(UP)/P(DOWN) far too sharp; that overconfidence is what pushed the
        development-fitted temperature to its upper bound. Folds are contiguous
        blocks of the (time-ordered) training rows, so a held-out block is
        always scored by a model that could not see it while still using only
        the training snapshot itself.
        """
        n=len(y)
        if n<folds*20:
            self.sigma_source="in_sample_small_n"
            return np.asarray(y,dtype=float)-self.routine.predict(X)
        edges=np.linspace(0,n,folds+1).astype(int);residual=np.empty(n,dtype=float)
        filled=np.zeros(n,dtype=bool);ok=True
        for a,b in zip(edges[:-1],edges[1:]):
            keep=np.ones(n,dtype=bool);keep[a:b]=False
            try:
                model=make_pipeline(SimpleImputer(strategy="constant",fill_value=0),StandardScaler(),
                                    Ridge(alpha=max(self.alpha,1e-6)))
                model.fit(X[keep],y[keep]);residual[a:b]=y[a:b]-model.predict(X[a:b]);filled[a:b]=True
            except Exception:
                ok=False;break
        if not ok or not filled.all():
            self.sigma_source="in_sample_fallback"
            return np.asarray(y,dtype=float)-self.routine.predict(X)
        self.sigma_source=f"out_of_fold_{folds}fold"
        return residual
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
        dev=[r for r in rows if r["cohort"]==cohort and r["label_available_date"]<TRAIN_END
             and np.isfinite(r.get("volatility_unit_pct",np.nan)) and r["volatility_unit_pct"]>0]
        z=np.asarray([abs(float(r["net_pct"]))/float(r["volatility_unit_pct"]) for r in dev])
        k=float(np.quantile(z,SIDEWAYS_TARGET_SHARE)) if len(z) else None
        labels=[]
        if k is not None:
            for row in rows:
                if row["cohort"]!=cohort:continue
                band=k*float(row.get("volatility_unit_pct",0.0))
                row["label_vol"]=direction_label(row["net_pct"],band) if band>0 else "SIDEWAYS"
                row["label_vol_band_pct"]=band
            labels=[r["label_vol"] for r in dev]
        counts={name:labels.count(name) for name in names}
        result[cohort]={"k":k,"development_n":len(dev),"development_class_counts":counts,
          "development_class_share_pct":{n:round(100*counts[n]/len(dev),3) if dev else None for n in names},
          "definition":"SIDEWAYS iff abs(net return after existing costs) <= k * signal-day 20d realized daily volatility * sqrt(h/20); UP/DOWN outside band",
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
    fit=minimize_scalar(objective,bounds=TEMPERATURE_LOG_BOUNDS,method="bounded",options={"xatol":1e-5})
    log_t=float(fit.x)
    at_bound=bool(log_t>=TEMPERATURE_LOG_BOUNDS[1]-1e-3 or log_t<=TEMPERATURE_LOG_BOUNDS[0]+1e-3)
    return {"temperature":float(np.exp(log_t)),"temperature_log":log_t,"fit_n":len(y),
            "status":"DEVELOPMENT_OOF_TEMPERATURE_SCALING",
            "temperature_log_bounds":list(TEMPERATURE_LOG_BOUNDS),
            "temperature_at_bound":at_bound,
            "development_oof_logloss_before":objective(0.),"development_oof_logloss_after":float(fit.fun)}

def _choose_development_k(rows):
    """Compatibility alias for callers/tests; returns frozen cohort settings."""
    return _freeze_volatility_labels(rows)

def _causal_class_priors(all_train,eval_rows,window_days=PRIOR_SHRINKAGE_WINDOW_DAYS,
                         half_life_days=None):
    """Per-signal-date class prior from labels already observable at that date.

    Only labels with ``label_available_date`` strictly before the signal date are
    counted, so the prior can be computed live. The window is a choice made on
    development rows only; when the window holds fewer than MIN_PRIOR_ROWS
    labels the expanding history before the signal date is used, and only when
    that is also too small does the prior fall back to uniform.

    ``half_life_days`` turns the flat window into an exponentially decayed one:
    a label that matured ``H`` days ago counts half as much as one that matured
    today. Since the decay factor separates into a per-row weight and a
    per-signal factor that cancels in the normalisation, the prior stays an
    exact causal quantity (labels before the signal date, nothing else) and a
    short half-life lets it follow a regime turn instead of averaging over it.
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
    prior_counts=onehot[order]
    if half_life_days:
        reference=date.fromisoformat(str(dates[-1]))
        age=np.asarray([(date.fromisoformat(str(d))-reference).days for d in dates],dtype=float)
        prior_counts=prior_counts*(2.0**(age/float(half_life_days)))[:,None]
    cumulative=np.vstack([np.zeros(3),np.cumsum(prior_counts,axis=0)])
    cache={};out=np.full((len(eval_rows),3),1/3,dtype=float)
    for i,row in enumerate(eval_rows):
        signal=str(row["signal_date"])
        if signal not in cache:
            floor=(date.fromisoformat(signal)-timedelta(days=int(window_days))).isoformat()
            low=int(np.searchsorted(dates,floor,"left"));high=int(np.searchsorted(dates,signal,"left"))
            counts=cumulative[high]-cumulative[low]
            if high-low<MIN_PRIOR_ROWS:
                counts=cumulative[high]
            cache[signal]=counts/counts.sum() if counts.sum()>0 else None
        value=cache[signal]
        if value is not None:
            out[i]=value
    return out

def _select_prior_half_life(all_train,eval_rows,grid=PRIOR_HALF_LIFE_GRID):
    """Pick the causal-prior memory on development log-loss only.

    The prior is a forecast of the realised class of each development row that
    uses nothing but labels observable before that row's signal date, so scoring
    it on development rows cannot leak. Validation and diagnostic rows only ever
    receive the half-life chosen here.
    """
    names=("DOWN","SIDEWAYS","UP")
    if not eval_rows:
        return None,{"status":"NO_DEVELOPMENT_ROWS"}
    y=np.asarray([names.index(r["label_vol"]) for r in eval_rows],dtype=int)
    best=(None,None);table={}
    for half_life in tuple(grid)+(None,):
        priors=_causal_class_priors(all_train,eval_rows,half_life_days=half_life)
        picked=np.clip(priors[np.arange(len(y)),y],1e-6,1.)
        logloss=float(-np.mean(np.log(picked)))
        argmax_accuracy=100*float(np.mean(priors.argmax(axis=1)==y))
        table["flat_250d" if half_life is None else f"half_life_{int(half_life)}d"]={
            "prior_log_loss_nats":logloss,"prior_argmax_accuracy_pct":argmax_accuracy}
        if best[1] is None or logloss<best[1]-1e-12:best=(half_life,logloss)
    return best[0],{"selected_half_life_days":best[0],"basis":"minimum development causal-prior log-loss",
        "candidates":table,"window_days":PRIOR_SHRINKAGE_WINDOW_DAYS}

def _class_prior(rows):
    names=("DOWN","SIDEWAYS","UP")
    if not rows:return {name:1/3 for name in names}
    counts={name:sum(1 for r in rows if r.get("label_vol")==name) for name in names}
    total=sum(counts.values())
    return {name:(counts[name]/total if total else 1/3) for name in names}

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
    """Argmax of p/q^tau (the PR #18 rule), kept for a like-for-like comparison.

    tau=0 reproduces the plain probability argmax. This rule divides by the
    causal prior alone even though the model already learned the training class
    prior, so on a development period with a different class mix it can look
    balanced while never expressing the model's own ranking differences.
    """
    tau=float(tau or 0.)
    if tau<=0:
        return np.asarray(probs,dtype=float).argmax(axis=1)
    adjusted=np.clip(np.asarray(probs,dtype=float),1e-9,1.)/\
        np.power(np.clip(np.asarray(priors,dtype=float),1e-3,1.),tau)
    return adjusted.argmax(axis=1)

def _prior_shift_decisions(probs,priors,train_prior,tau):
    """Argmax of p * (causal_prior / training_prior)^tau.

    This is the standard label-shift correction: the model's probabilities
    already carry the training class mix, so only the *change* in the class mix
    between training and the signal date may move a decision. It is a
    multiplicative reweighting, so the within-row ordering of the model's
    likelihood ratios is preserved instead of being averaged away by a mixture.
    tau=0 is the plain argmax of the calibrated probabilities.
    """
    tau=float(tau or 0.)
    p=np.asarray(probs,dtype=float)
    if tau<=0:
        return p.argmax(axis=1)
    base=np.asarray(train_prior,dtype=float)
    if base.ndim==1:
        base=base.reshape(1,-1)
    elif base.shape[0]!=len(p):
        base=base.reshape(-1)[:3].reshape(1,3)
    base=np.clip(base,1e-6,1.)
    now=np.clip(np.asarray(priors,dtype=float),1e-6,1.)
    return (p*np.power(now/base,tau)).argmax(axis=1)

def _fit_decision_tau(rows,probs,priors,train_prior=None):
    """Decision threshold chosen on development accuracy only."""
    names=("DOWN","SIDEWAYS","UP")
    y=np.asarray([names.index(r["label_vol"]) for r in rows],dtype=int)
    if train_prior is None:
        train_prior=np.asarray([_class_prior(rows)[n] for n in names],dtype=float)
    elif isinstance(train_prior,dict):
        train_prior=np.asarray([float(train_prior.get(n,1/3)) for n in names],dtype=float)
    else:
        train_prior=np.asarray(train_prior,dtype=float).reshape(-1)
    if train_prior.shape!=(3,):
        train_prior=np.asarray([_class_prior(rows)[n] for n in names],dtype=float)
    best=(-1.,0.)
    for tau in DECISION_TAU_GRID:
        accuracy=float(np.mean(_prior_shift_decisions(probs,priors,train_prior,tau)==y))
        if accuracy>best[0]+1e-12:best=(accuracy,float(tau))
    return best[1],best[0]

def _fit_decision_tau_prior_divide(rows,probs,priors):
    """Same grid for the PR #18 rule, so the two decision rules are compared."""
    names=("DOWN","SIDEWAYS","UP")
    y=np.asarray([names.index(r["label_vol"]) for r in rows],dtype=int)
    best=(-1.,0.)
    for tau in DECISION_TAU_GRID:
        accuracy=float(np.mean(_prior_corrected_decisions(probs,priors,tau)==y))
        if accuracy>best[0]+1e-12:best=(accuracy,float(tau))
    return best[1],best[0]

def _stance_dispersion(probs):
    """How far the model's UP-minus-DOWN stance actually varies row to row."""
    p=np.asarray(probs,dtype=float)
    if not len(p):return {"n":0}
    stance=p[:,2]-p[:,0]
    q=np.percentile(stance,[10,50,90])
    return {"n":int(len(p)),"p_up_minus_p_down_p10_p50_p90":[float(q[0]),float(q[1]),float(q[2])],
        "stance_iqr":float(np.percentile(stance,75)-np.percentile(stance,25)),
        "stance_std":float(np.std(stance))}

def _decision_diversity(predictions):
    """Distribution of the decision itself, with an explicit collapse label."""
    names=("DOWN","SIDEWAYS","UP")
    pred=np.asarray(predictions,dtype=int)
    n=len(pred)
    if not n:return {"n":0,"collapse_status":"NO_ROWS"}
    counts={name:int(np.sum(pred==j)) for j,name in enumerate(names)}
    shares={name:(100*counts[name]/n) for name in names}
    largest=max(shares.values())
    if largest>=100*DECISION_PINNED_SHARE:
        status="ONE_CLASS_PINNED"
    elif largest>=100*DECISION_DOMINANT_SHARE:
        status="ONE_CLASS_DOMINANT"
    else:
        status="DISTRIBUTED"
    entropy=float(-sum((c/n)*math.log(c/n) for c in counts.values() if c))
    return {"n":n,"predicted_counts":counts,"predicted_share_pct":shares,
        "largest_class_share_pct":largest,"distinct_classes":int(sum(c>0 for c in counts.values())),
        "prediction_entropy_nats":entropy,"collapse_status":status}

def _finite(value):
    try:value=float(value)
    except (TypeError,ValueError):return float("nan")
    return value if np.isfinite(value) else float("nan")

def _regime_off(row,params):
    """Risk-off trend state from the row's own signal-day features.

    Every input is computed through the signal-day close only: the cohort index
    features `market_index_ma200_gap_pct` (index versus its 200-day mean),
    `market_drawdown_120_pct` (index versus its 120-day high) and
    `market_index_return_20d_pct`, plus the instrument's own `ma200_gap` and
    `ret20`. Nothing later than the signal date is read, and no other row is
    consulted. A missing input makes the state False rather than guessing.
    """
    if not params or params.get("mode") in (None,"none"):return False
    def value(key):
        try:number=float(row.get(key))
        except (TypeError,ValueError):return float("nan")
        return number if math.isfinite(number) else float("nan")
    state=str(params.get("state","index"))
    weak_ret=float(params["index_return_20d_max_pct"])
    gap_max=float(params["ma200_gap_max_pct"])
    if state=="instrument":
        stock_ret=value("ret20");stock_gap=value("ma200_gap")
        if not (math.isfinite(stock_ret) and math.isfinite(stock_gap)):return False
        return stock_ret<=weak_ret and stock_gap<=gap_max
    index_ret=value("market_index_return_20d_pct")
    if not math.isfinite(index_ret) or index_ret>weak_ret:return False
    broken=[]
    index_gap=value("market_index_ma200_gap_pct")
    if math.isfinite(index_gap):broken.append(index_gap<=gap_max)
    index_drawdown=value("market_drawdown_120_pct")
    if math.isfinite(index_drawdown):broken.append(index_drawdown<=float(params["drawdown_120_min_pct"]))
    if not (broken and any(broken)):return False
    if state=="index_and_instrument":
        stock_gap=value("ma200_gap")
        return math.isfinite(stock_gap) and stock_gap<=0.
    return True

def _apply_regime_gate(rows,probs,predictions,params):
    """Move UP decisions to a conservative class while the index trend is risk-off.

    Returns (gated_predictions, active_mask, override_mask). Probabilities are
    untouched: the gate acts on the decision only, so its effect on Brier and on
    the probability diagnostics is exactly zero by construction.
    """
    pred=np.asarray(predictions,dtype=int).copy()
    n=len(pred)
    active=np.asarray([_regime_off(row,params) for row in rows],dtype=bool) if n else np.zeros(0,dtype=bool)
    override=np.zeros(n,dtype=bool)
    if not n or not params or params.get("mode") in (None,"none"):
        return pred,active,override
    override=active&(pred==2)
    if override.any():
        target=str(params.get("target","DOWN")).upper()
        if target=="SECOND_CHOICE":
            p=np.asarray(probs,dtype=float)
            pick=np.where(p[:,0]>=p[:,1],0,1)
            pred[override]=pick[override]
        else:
            pred[override]=1 if target.startswith("SIDE") else 0
    return pred,active,override

def _macro_f1_predictions(y,pred):
    scores=[]
    for klass in range(3):
        tp=float(np.sum((pred==klass)&(y==klass)))
        predicted=float(np.sum(pred==klass));actual=float(np.sum(y==klass))
        precision=tp/predicted if predicted else 0.
        recall=tp/actual if actual else 0.
        scores.append(2*precision*recall/(precision+recall) if precision+recall else 0.)
    return sum(scores)/len(scores)

def _fit_regime_gate(rows,probs,predictions):
    """Fit the trend gate on development OOF decisions only.

    The objective is development three-way accuracy of the resulting decision,
    under three guards: the gate must override a meaningful number of development
    rows, it may not pin the decision to fewer than two classes, and it may not
    drop macro-F1 by more than REGIME_GATE_MAX_MACRO_F1_DROP_PP (so trading an UP
    pin for a DOWN pin is rejected). Without a positive development gain the gate
    stays off (`mode: none`) and the decision is the ungated one.
    """
    names=("DOWN","SIDEWAYS","UP")
    pred=np.asarray(predictions,dtype=int)
    if not len(rows):
        return {"chosen":{"mode":"none","reason":"NO_ROWS"},
                "selection":{"status":"NO_ROWS","candidates":[]}}
    y=np.asarray([names.index(r["label_vol"]) for r in rows],dtype=int)
    base_accuracy=float(np.mean(pred==y));base_f1=_macro_f1_predictions(y,pred)
    index_gap=np.asarray([_finite(r.get("market_index_ma200_gap_pct")) for r in rows],dtype=float)
    drawdown=np.asarray([_finite(r.get("market_drawdown_120_pct")) for r in rows],dtype=float)
    index_ret=np.asarray([_finite(r.get("market_index_return_20d_pct")) for r in rows],dtype=float)
    stock_gap=np.asarray([_finite(r.get("ma200_gap")) for r in rows],dtype=float)
    stock_ret=np.asarray([_finite(r.get("ret20")) for r in rows],dtype=float)
    up=pred==2
    p=np.asarray(probs,dtype=float)
    second=np.where(p[:,0]>=p[:,1],0,1)
    candidates=[]
    for state in REGIME_GATE_STATES:
        drawdown_grid=REGIME_GATE_DRAWDOWN_GRID if state!="instrument" else (REGIME_GATE_DRAWDOWN_GRID[0],)
        for gap_max in REGIME_GATE_MA200_GAP_GRID:
            for drawdown_min in drawdown_grid:
                for ret_max in REGIME_GATE_INDEX_RET20_GRID:
                    if state=="instrument":
                        active=np.isfinite(stock_ret)&(stock_ret<=ret_max)&\
                            np.isfinite(stock_gap)&(stock_gap<=gap_max)
                    else:
                        trend_broken=np.zeros(len(rows),dtype=bool)
                        if np.isfinite(index_gap).any():trend_broken|=np.isfinite(index_gap)&(index_gap<=gap_max)
                        if np.isfinite(drawdown).any():trend_broken|=np.isfinite(drawdown)&(drawdown<=drawdown_min)
                        active=np.isfinite(index_ret)&(index_ret<=ret_max)&trend_broken
                        if state=="index_and_instrument":
                            active&=np.isfinite(stock_gap)&(stock_gap<=0.)
                    override=active&up
                    overrides=int(override.sum())
                    if not overrides:
                        continue
                    for target in REGIME_GATE_TARGETS:
                        gated=pred.copy()
                        if target=="SECOND_CHOICE":gated[override]=second[override]
                        else:gated[override]=1 if target=="SIDEWAYS" else 0
                        accuracy=float(np.mean(gated==y));f1=_macro_f1_predictions(y,gated)
                        counts={names[j]:int(np.sum(gated==j)) for j in range(3)}
                        candidates.append({"state":state,
                            "ma200_gap_max_pct":float(gap_max),
                            "drawdown_120_min_pct":float(drawdown_min),
                            "index_return_20d_max_pct":float(ret_max),"target":target,
                            "overrides_n":overrides,"activated_rows_n":int(active.sum()),
                            "activation_share_pct":100*float(active.mean()),
                            "accuracy_pct":100*accuracy,"accuracy_gain_pp":100*(accuracy-base_accuracy),
                            "macro_f1_pct":100*f1,"macro_f1_delta_pp":100*(f1-base_f1),
                            "predicted_counts":counts,
                            "distinct_classes":int(sum(1 for c in counts.values() if c))})
    eligible=[c for c in candidates if c["overrides_n"]>=REGIME_GATE_MIN_DEV_OVERRIDES
              and c["distinct_classes"]>=2
              and c["macro_f1_delta_pp"]>=-REGIME_GATE_MAX_MACRO_F1_DROP_PP]
    improving=[c for c in eligible if c["accuracy_gain_pp"]>=REGIME_GATE_MIN_ACCURACY_GAIN_PP]
    chosen=max(improving,key=lambda c:(c["accuracy_pct"],c["macro_f1_pct"],-c["overrides_n"])) if improving else None
    top=sorted(candidates,key=lambda c:(c["accuracy_pct"],c["macro_f1_pct"]),reverse=True)[:12]
    selection={"status":("DEVELOPMENT_GAIN_FOUND" if chosen else "NO_DEVELOPMENT_GAIN"),
        "scope":"development OOF decisions only; validation and diagnostic never enter this fit",
        "constraints":{"min_development_overrides":REGIME_GATE_MIN_DEV_OVERRIDES,
            "min_accuracy_gain_pp":REGIME_GATE_MIN_ACCURACY_GAIN_PP,
            "max_macro_f1_drop_pp":REGIME_GATE_MAX_MACRO_F1_DROP_PP,
            "require_at_least_two_predicted_classes":True,
            "require_positive_accuracy_gain":True},
        "objective":"maximum development three-way accuracy; ties -> macro-F1, then fewer overrides",
        "candidate_count":len(candidates),
        "base_development_accuracy_pct":100*base_accuracy,
        "base_development_macro_f1_pct":100*base_f1,
        "grid":{"state":list(REGIME_GATE_STATES),
            "ma200_gap_max_pct":list(REGIME_GATE_MA200_GAP_GRID),
            "drawdown_120_min_pct":list(REGIME_GATE_DRAWDOWN_GRID),
            "index_return_20d_max_pct":list(REGIME_GATE_INDEX_RET20_GRID),
            "target":list(REGIME_GATE_TARGETS)},
        "gate_features":list(REGIME_GATE_FEATURES),
        "candidates":candidates,"top_candidates":top}
    if chosen is None:
        return {"chosen":{"mode":"none",
            "reason":"NO_DEVELOPMENT_GAIN_UNDER_GUARDS","fitted_on":"development_only"},
            "selection":selection}
    params={"mode":"trend_override","state":chosen["state"],
        "ma200_gap_max_pct":chosen["ma200_gap_max_pct"],
        "drawdown_120_min_pct":chosen["drawdown_120_min_pct"],
        "index_return_20d_max_pct":chosen["index_return_20d_max_pct"],
        "target":chosen["target"],"fitted_on":"development_out_of_fold_decisions_only",
        "development_accuracy_pct":chosen["accuracy_pct"],
        "development_accuracy_gain_pp":chosen["accuracy_gain_pp"],
        "development_macro_f1_delta_pp":chosen["macro_f1_delta_pp"],
        "development_overrides_n":chosen["overrides_n"],
        "rule":("risk_off per the fitted state (index 20-day return weak and the longer trend broken, "
            "optionally requiring the instrument itself below its own 200-day mean); "
            "an UP decision under risk_off becomes target")}
    return {"chosen":params,"selection":selection}

def _fit_calibration(rows,probs,priors,train_prior=None):
    """Development-only probability calibration, separate from the decision.

    Two products with two jobs:

    * probability: development-OOF temperature scaling (fitted on log-loss).
    * decision: a label-shift rule argmax(p * (causal_prior/train_prior)**tau)
      with tau fitted on development three-way accuracy.

    The PR #18 pipeline mixed the probability toward the causal prior and then
    divided the mixture by that same prior, so the two corrections nearly
    cancelled while the mixture compressed the model's per-row differences to
    a few hundredths. Both effects are removed here; the mixture is reported as
    a reference (its development Brier gain and its dispersion cost) so the
    trade-off stays visible.
    """
    base=_fit_temperature(rows,probs)
    temperature=float(base.get("temperature",1.0))
    scaled=_apply_temperature(probs,temperature)
    names=("DOWN","SIDEWAYS","UP")
    train_prior=train_prior if train_prior is not None else _class_prior(rows)
    if isinstance(train_prior,dict):
        train_prior=np.asarray([float(train_prior.get(name,1/3)) for name in names],dtype=float)
    else:
        train_prior=np.asarray(train_prior,dtype=float).reshape(-1)
    if train_prior.shape!=(3,):
        train_prior=np.asarray([_class_prior(rows)[name] for name in names],dtype=float)
    y=np.asarray([names.index(r["label_vol"]) for r in rows],dtype=int)
    tau,accuracy=_fit_decision_tau(rows,scaled,priors,train_prior)
    tau_divide,accuracy_divide=_fit_decision_tau_prior_divide(rows,scaled,priors)
    predictions=_prior_shift_decisions(scaled,priors,train_prior,tau)
    weight=_fit_prior_mix(rows,scaled,priors)
    reference_weight=float(min(max(weight,0.),MAX_REFERENCE_PRIOR_MIX))
    reference_mixed=_apply_prior_mix(scaled,priors,reference_weight)
    result=dict(base)
    result.update({"prior_window_days":PRIOR_SHRINKAGE_WINDOW_DAYS,
        "decision_tau":float(tau),
        "decision_tau_prior_divide":float(tau_divide),
        "decision_rule":"argmax(p * (causal_prior / development_prior)**tau)",
        "decision_rule_prior_divide":"argmax(p / causal_prior**tau) [PR #18 rule, comparison only]",
        "train_prior":{name:float(train_prior[j]) for j,name in enumerate(names)},
        "probability_mixing":"NOT_APPLIED: the prior mixture feeds only the reference Brier below",
        "prior_mix":0.,
        "reference_prior_mix":reference_weight,
        "reference_prior_mix_uncapped":float(weight),
        "reference_prior_mix_cap":MAX_REFERENCE_PRIOR_MIX,
        "development_brier_temperature_only":_multiclass_brier(rows,scaled),
        "development_brier_reference_prior_mix":_multiclass_brier(rows,reference_mixed),
        "development_accuracy_argmax_pct":100*float(np.mean(scaled.argmax(axis=1)==y)),
        "development_accuracy_after_decision_pct":100*float(accuracy),
        "development_accuracy_prior_divide_pct":100*float(accuracy_divide),
        "development_dispersion_temperature_only":_stance_dispersion(scaled),
        "development_dispersion_reference_prior_mix":_stance_dispersion(reference_mixed),
        "development_decision_diversity":_decision_diversity(predictions),
        "scope":"development out-of-fold predictions only; temperature, prior half-life and tau are applied unchanged to validation and diagnostic"})
    result["development_brier_reference_gain"]=result["development_brier_temperature_only"]-\
        result["development_brier_reference_prior_mix"]
    if result.get("status")=="INSUFFICIENT_DEV_OOF_FOR_CALIBRATION":
        result.update({"decision_tau":0.,"decision_tau_prior_divide":0.,"reference_prior_mix":0.})
    return result

def _ece(y,p,klass):
    confidence=p[:,klass]; actual=(y==klass).astype(float); ece=0.; bins=[]
    for b in range(10):
        lo=b/10; hi=(b+1)/10; mask=(confidence>=lo)&((confidence<hi) if b<9 else (confidence<=hi))
        if mask.any():
            conf=float(confidence[mask].mean()); acc=float(actual[mask].mean()); ece+=mask.sum()/len(y)*abs(conf-acc)
            bins.append({"bin":f"{lo:.1f}-{hi:.1f}","n":int(mask.sum()),"confidence":conf,"observed":acc})
    return {"ece":ece,"bins":bins}

def _metrics(rows,probs,theta=.5,label_key=None,predictions=None):
    names=("DOWN","SIDEWAYS","UP"); label_key=label_key or f"label_{theta}"; y=np.asarray([names.index(r[label_key]) for r in rows])
    pred=(np.asarray(predictions,dtype=int) if predictions is not None else probs.argmax(axis=1))
    cm=confusion_matrix(y,pred,labels=[0,1,2]);pr,re,fs,_=precision_recall_fscore_support(y,pred,labels=[0,1,2],zero_division=0)
    briers=[];eces={}
    for j,name in enumerate(names):
        actual=(y==j).astype(float); briers.append(float(np.mean((probs[:,j]-actual)**2))); eces[name]=_ece(y,probs,j)
    class_counts=np.bincount(y,minlength=3);priors=class_counts/max(1,len(y))
    prior_entropy=float(-np.sum(priors[priors>0]*np.log(priors[priors>0])))
    out={"n":len(rows),"label_key":label_key,"class_distribution":{names[j]:int(class_counts[j]) for j in range(3)},
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
        v=spearmanr(up_score[indexes],[rows[i]["net_pct"] for i in indexes]).statistic
        if np.isfinite(v):daily_ic.append(float(v))
    order=np.argsort(up_score,kind="mergesort");k=max(1,int(np.ceil(len(rows)*.15)));top=order[-k:]
    out["datewise_spearman_mean"]=float(np.mean(daily_ic)) if daily_ic else None
    out["datewise_spearman_n_days"]=len(daily_ic)
    out["top15_n"]=len(top);out["top15_win_rate_pct"]=100*float(np.mean([rows[i]["net_pct"]>0 for i in top]))
    out["top15_mean_net_pct"]=float(np.mean([rows[i]["net_pct"] for i in top]))
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
            if kind=="band_z":
                target=np.asarray([float(r["net_pct"])/_volatility_unit(r) for r in past])
                bands=np.asarray([_band_z(eval_rows[i]) for i in idx])
            else:
                target=np.asarray([float(r["net_pct"]) for r in past])
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
    raw_probs=probs.copy()
    # A development call that fits its own calibration also owns the regime-gate
    # fit; calls that receive an existing temperature_info never refit it, so the
    # gate params reach validation and diagnostic unchanged.
    owns_calibration=period=="development" and temperature_info is None
    if owns_calibration:
        half_life,half_life_report=_select_prior_half_life(all_train,eval_rows)
        priors=_causal_class_priors(all_train,eval_rows,half_life_days=half_life)
        temperature_info=_fit_calibration(eval_rows,probs,priors,_class_prior(eval_rows))
        temperature_info["prior_half_life_days"]=half_life
        temperature_info["prior_half_life_selection"]=half_life_report
    else:
        half_life=(temperature_info or {}).get("prior_half_life_days")
        priors=_causal_class_priors(all_train,eval_rows,half_life_days=half_life)
    temperature=float((temperature_info or {}).get("temperature",1.0))
    probs=_apply_temperature(raw_probs,temperature)
    train_prior=(temperature_info or {}).get("train_prior") or _class_prior(eval_rows)
    train_prior_array=np.asarray([float(train_prior.get(name,1/3)) for name in classes],dtype=float)
    decision_tau=float((temperature_info or {}).get("decision_tau",0.) or 0.)
    decision_predictions=_prior_shift_decisions(probs,priors,train_prior_array,decision_tau)
    prior_divide_tau=float((temperature_info or {}).get("decision_tau_prior_divide",0.) or 0.)
    prior_divide_predictions=_prior_corrected_decisions(probs,priors,prior_divide_tau)
    # Regime/trend gate: a causal index-state rule that may turn an UP decision
    # into DOWN/SIDEWAYS. Fitted on this development sample only.
    gate_info=(temperature_info or {}).get("regime_gate") or {"mode":"none"}
    gate_selection=(temperature_info or {}).get("regime_gate_selection")
    if owns_calibration:
        gate_fit=_fit_regime_gate(eval_rows,probs,decision_predictions)
        gate_info=gate_fit["chosen"]
        gate_selection=gate_fit["selection"]
        temperature_info["regime_gate"]=gate_info
        temperature_info["regime_gate_selection"]=gate_selection
    gated_predictions,gate_active,gate_override=_apply_regime_gate(
        eval_rows,probs,decision_predictions,gate_info)
    gate_override_n=int(gate_override.sum())
    payoffs=payoffs or (_class_payoffs(eval_rows) if period=="development" else {"DOWN":-1.,"SIDEWAYS":0.,"UP":1.})
    theta_results={}
    for theta in THETAS:
        fixed=_metrics(eval_rows,probs,theta)
        theta_results[str(theta)]={"metrics":fixed}
    vol_metrics=_metrics(eval_rows,probs,label_key="label_vol",predictions=decision_predictions)
    target_actual=np.asarray([int(r.get("target_touch_theta_0.5")=="TARGET_FIRST") for r in eval_rows],dtype=float)
    target_report={"n":len(eval_rows),"observed_target_first_pct":100*float(target_actual.mean()),
      "mean_predicted_probability_pct":100*float(target_probs.mean()),"brier":float(np.mean((target_probs-target_actual)**2))}
    vol_metrics["target_hit_probability"]=target_report
    vol_metrics["causal_prior_baseline"]=_causal_prior_baseline(all_train,eval_rows)
    vol_metrics["probability_skill"]=_probability_skill(eval_rows,probs,priors)
    vol_metrics["decision"]=_balanced_decision_report(eval_rows,decision_predictions,priors,decision_tau,
        rule=(temperature_info or {}).get("decision_rule","argmax(p * (causal_prior / development_prior)**tau)"))
    vol_metrics["plain_argmax_decision"]=_balanced_decision_report(eval_rows,probs.argmax(axis=1),priors,0.)
    vol_metrics["prior_divide_decision"]=_balanced_decision_report(eval_rows,prior_divide_predictions,priors,
        prior_divide_tau,rule="argmax(p / causal_prior**tau) [PR #18 rule]")
    vol_metrics["regime_gate_decision"]=_balanced_decision_report(eval_rows,gated_predictions,priors,decision_tau,
        rule=(gate_info or {}).get("rule","regime trend gate on the label-shift decision"))
    gated_counts={classes[j]:int(np.sum(gated_predictions==j)) for j in range(3)}
    vol_metrics["regime_gate"]={"params":gate_info,
        "gate_features":list(REGIME_GATE_FEATURES),
        "activated_rows_n":int(gate_active.sum()),
        "activated_share_pct":100*float(gate_active.mean()) if len(gate_active) else 0.,
        "overrides_n":gate_override_n,
        "overrides_share_pct":100*gate_override_n/len(eval_rows) if eval_rows else 0.,
        "predicted_counts":gated_counts,
        "accuracy_gain_vs_ungated_pp":(100*float(np.mean(gated_predictions==np.asarray(
            [classes.index(r["label_vol"]) for r in eval_rows],dtype=int)))
            -float(vol_metrics["decision"].get("accuracy_pct") or 0.)),
        "scope":"development OOF fit only" if owns_calibration else "fitted parameters reused unchanged from development",
        "selection":gate_selection if owns_calibration else None}
    # Where does the collapse start: the model itself or the calibration? The raw
    # argmax is the out-of-fold model output before temperature and decision.
    vol_metrics["raw_argmax_decision"]=_balanced_decision_report(eval_rows,raw_probs.argmax(axis=1),priors,0.,
        rule="argmax(raw development-OOF model probabilities)")
    vol_metrics["probability_dispersion"]={
        "raw":_stance_dispersion(raw_probs),"calibrated":_stance_dispersion(probs),
        "raw_brier":_multiclass_brier(eval_rows,raw_probs),
        "calibrated_brier":_multiclass_brier(eval_rows,probs),
        "reference_prior_mix_dispersion":_stance_dispersion(_apply_prior_mix(
            probs,priors,float((temperature_info or {}).get("reference_prior_mix",0.) or 0.)))}
    policies=_policy_table(eval_rows,probs,payoffs)
    result={"n":len(eval_rows),"metrics":vol_metrics,"fixed_band_metrics":{k:v["metrics"] for k,v in theta_results.items()},
      "theta_results":theta_results,"buy_policy_candidates":policies,"buy_policy":None,
      "temperature":temperature_info,"class_payoffs_from_development":payoffs,
      "policy_selection_scope":"development only; thresholds are spread, sideways-probability, and net EV after existing costs",
      "probability_sample":{"p_up_p10_p50_p90":np.percentile(probs[:,2],[10,50,90]).tolist(),
        "p_down_p10_p50_p90":np.percentile(probs[:,0],[10,50,90]).tolist(),
        "p_sideways_p10_p50_p90":np.percentile(probs[:,1],[10,50,90]).tolist()},
      "decision_summary":{"rule":(temperature_info or {}).get("decision_rule"),"tau":decision_tau,
        "prior_half_life_days":(temperature_info or {}).get("prior_half_life_days"),
        **_decision_diversity(decision_predictions)},
      "gated_decision_summary":{"rule":(gate_info or {}).get("rule"),
        "gate_mode":(gate_info or {}).get("mode"),"overrides_n":gate_override_n,
        **_decision_diversity(gated_predictions)}}
    if include_samples:
        result["evaluation_samples"]=[{"label":row["label_vol"],
            "prediction":classes[int(decision)],
            "gated_prediction":classes[int(gated)],
            "regime_gate_active":bool(active),
            "regime_gate_override":bool(override),
            "argmax_prediction":classes[int(np.argmax(prob))],
            "probabilities":{"DOWN":float(prob[0]),"SIDEWAYS":float(prob[1]),"UP":float(prob[2])},
            "raw_probabilities":{"DOWN":float(raw[0]),"SIDEWAYS":float(raw[1]),"UP":float(raw[2])},
            "causal_prior":{"DOWN":float(prior[0]),"SIDEWAYS":float(prior[1]),"UP":float(prior[2])},
            "decision_rule":(temperature_info or {}).get("decision_rule"),
            "market_index_ma200_gap_pct":row.get("market_index_ma200_gap_pct"),
            "market_index_return_20d_pct":row.get("market_index_return_20d_pct"),
            "earnings_momentum":row.get("earnings_momentum")}
            for row,prob,raw,prior,decision,active,override,gated in zip(
                eval_rows,probs,raw_probs,priors,decision_predictions,gate_active,gate_override,gated_predictions)]
    return result

def _balanced_decision_report(rows,predictions,priors,tau,rule="argmax(p / causal_prior**tau)"):
    """Accuracy and class spread of one decision rule on the same rows."""
    names=("DOWN","SIDEWAYS","UP")
    if not len(rows or []):
        return {"n":0,"collapsed":False,"collapse_status":"NO_ROWS"}
    y=np.asarray([names.index(r["label_vol"]) for r in rows],dtype=int)
    predictions=np.asarray(predictions,dtype=int)
    diversity=_decision_diversity(predictions)
    out={"n":len(y),"tau":float(tau),"rule":rule,
         "predicted_counts":{name:int(np.sum(predictions==j)) for j,name in enumerate(names)},
         "predicted_share_pct":diversity["predicted_share_pct"],
         "largest_class_share_pct":diversity["largest_class_share_pct"],
         "collapse_status":diversity["collapse_status"],
         "collapsed":diversity["collapse_status"] in ("ONE_CLASS_PINNED","ONE_CLASS_DOMINANT"),
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
    volatility_k=_freeze_volatility_labels(rows+special_eval.get("US",[])+special_eval.get("TIGER_ETF",[]))
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
      "volatility_band":{"formula":"k_cohort * signal_day_realized_vol20_pct * sqrt(h/20)","target_development_sideways_share_pct":100*SIDEWAYS_TARGET_SHARE,
        "cohort_parameters":volatility_k}},
      "direction3":{"label_definition":"net return after existing cost; volatility-scaled symmetric SIDEWAYS band; cohort k fitted using only outcomes available before TRAIN_END","cohort_k":volatility_k,"probability_calibration":{},"policy_results":{},
        "minimum_development_ic_for_edge_label":MIN_DEVELOPMENT_IC},
      "extended_pool_collection":extended_pool_collection,
      "minimum_development_ic_for_edge_label":MIN_DEVELOPMENT_IC,
      "selection_rule":"Development-only walk-forward: select one model/horizon by mean datewise Spearman(P(UP), realized net return), requiring positive IC, at least 100 top-15% development candidates, a development three-way accuracy at least the live-computable causal class prior, and a decision that names at least two classes on development. Model grid covers multinomial logistic, random forest, band regression (predicted net return mapped through the frozen volatility band), a winsorised/MAD variant and a volatility-normalised (z-return) variant. Probabilities are calibrated on development OOF with temperature scaling only; the class prior is applied at the decision stage as a label-shift correction argmax(p * (causal_prior/development_prior)**tau), and the causal-prior memory (exponential half-life) and tau are both chosen on development only. A causal regime/trend gate (signal-day cohort-index 20-day return, 200-day gap and 120-day drawdown, with an optional instrument-trend confirmation) may then move an UP decision to DOWN/SIDEWAYS; its state, thresholds and target class are fitted on development OOF decisions only, with minimum-override, two-class and macro-F1 guards. Select spread + sideways-probability + net expected-value policy on development only; all selected policy OOS results are reported.",
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
        chosen=max(eligible_choices,default=(-1e9,None,None))
        chosen_ic=((development_cache[(chosen[1],chosen[2])].get("metrics") or {}).get("datewise_spearman_mean")
                   if chosen[1] is not None else None)
        cohort_report={"selected_model_development_only":chosen[1],"selected_horizon_development_only":chosen[2],
          "training_pool_scope":"available full ordinary-US equity cohort; SOXL is evaluation-only" if cohort=="US" else "63 audit cohort for selection; full cohort reported as separate training comparison",
          "selected_development_datewise_spearman":chosen_ic,
          "model_selection_status":_edge_label_status(chosen[1] is not None,chosen_ic),
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
                  "probability_sample":raw.get("probability_sample"),
                  "decision_summary":raw.get("decision_summary")}
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
    out["direction3"]["calibration_scope"]=("development OOF probabilities only: one temperature per selected model/horizon/training universe "
        f"(log-space bounds {list(TEMPERATURE_LOG_BOUNDS)}), one causal-prior memory (exponential half-life from labels observable before each signal date, window "
        f"{PRIOR_SHRINKAGE_WINDOW_DAYS} days) and one label-shift decision exponent, each chosen on development data and applied unchanged to validation and diagnostic. "
        "The probability-level prior mixture is reported as a reference only and does not feed the decision. "
        "The regime/trend gate is fitted on development OOF decisions only (state plus thresholds on the signal-day cohort-index 20-day return, 200-day gap and 120-day drawdown, and the conservative target class) and is applied unchanged out of sample; it never changes a probability.")
    out["direction3"]["probability_calibration"]={cohort:value.get("calibration_by_universe",{})
        for cohort,value in out["direction3"].items() if isinstance(value,dict) and "universe_comparison" in value}
    out["direction3"]["policy_results"]={cohort:{universe_name:periods for universe_name,periods in value.get("universe_comparison",{}).items()}
        for cohort,value in out["direction3"].items() if isinstance(value,dict) and "universe_comparison" in value}
    out["direction3"]["label_definition_comparison"]={cohort:{"volatility_band":v,
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
