"""Universe expansion and causal three-way direction research for audit-63.

Research-only. Signals are observed at the close; entries use the next open.
Labels are revealed only after their horizon close and models refit annually
using labels available strictly before the signal date.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import date
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
    confusion_matrix, f1_score, log_loss, precision_recall_fscore_support,
    roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from providers import get_historical_prices
from universe import get_universe, MANDATORY_KR, MANDATORY_US
from engine import compute

TRAIN_END="2025-11-01"; VALID_END="2026-04-01"; HOLDOUT_END="2026-11-01"
SAMPLE_START="2016-01-01"; COST_PCT=.35; THETAS=(.3,.5,1.0)
HORIZONS=(5,10,20); MIN_TRAIN=500; REFIT_MONTH=1; SIGNAL_STEP=5
FEATURES=("ret5","ret20","ret60","ret120","atr_pct","vol20_pct",
          "rsi","ma20_gap","ma60_gap","ma200_gap","turnover_ratio",
          "volume_flow5","gap_pct","intraday_strength20","xs_ret20",
          "xs_ret60","xs_volatility20","xs_turnover20")
CENTERS={"ret5":0,"ret20":0,"ret60":0,"ret120":0,"atr_pct":3,"vol20_pct":3,
 "rsi":50,"ma20_gap":0,"ma60_gap":0,"ma200_gap":0,"turnover_ratio":1,
 "volume_flow5":0,"gap_pct":0,"intraday_strength20":.5,"xs_ret20":.5,
 "xs_ret60":.5,"xs_volatility20":.5,"xs_turnover20":.5}
SCALES={"ret5":10,"ret20":20,"ret60":35,"ret120":55,"atr_pct":3,"vol20_pct":5,
 "rsi":25,"ma20_gap":10,"ma60_gap":18,"ma200_gap":30,"turnover_ratio":2,
 "volume_flow5":1,"gap_pct":5,"intraday_strength20":.5,"xs_ret20":.5,
 "xs_ret60":.5,"xs_volatility20":.5,"xs_turnover20":.5}

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
    rows={c:[(ticker,name,market,sector) for ticker,name,market,sector in items] for c,items in rows.items()}
    rows["universe_sources"]={k:universe.get(k+"_source") for k in ("KR","KQ","US")}
    rows["universe_errors"]={k:universe.get(k+"_error") for k in ("KR","KQ") if universe.get(k+"_error")}
    return rows

def _audit_symbols(universe):
    return {"KOSPI":{x[0] for x in universe.get("KOSPI",[])[:30]},
            "KOSDAQ":{x[0] for x in universe.get("KOSDAQ",[])[:30]},
            "US":{"SOXL"},"TIGER_ETF":{x[0] for x in MANDATORY_KR}}

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

def _active_features(rows):
    return tuple(k for k in FEATURES if np.std([float(r.get(k,CENTERS[k])) for r in rows])>1e-12)

def _features(rows,active=None):
    active=FEATURES if active is None else active
    return np.asarray([[np.clip((float(r.get(k,CENTERS[k]))-CENTERS[k])/SCALES[k],-5,5) for k in active]
                       for r in rows],dtype=np.float32)

def _model(kind):
    if kind=="multinomial_logistic":
        return make_pipeline(SimpleImputer(strategy="constant",fill_value=0),StandardScaler(),
             LogisticRegression(C=.15,max_iter=300,class_weight="balanced",random_state=3610))
    return RandomForestClassifier(n_estimators=80,max_depth=5,min_samples_leaf=20,
            class_weight="balanced_subsample",random_state=3610,n_jobs=1)

def _ece(y,p,klass):
    confidence=p[:,klass]; actual=(y==klass).astype(float); ece=0.; bins=[]
    for b in range(10):
        lo=b/10; hi=(b+1)/10; mask=(confidence>=lo)&((confidence<hi) if b<9 else (confidence<=hi))
        if mask.any():
            conf=float(confidence[mask].mean()); acc=float(actual[mask].mean()); ece+=mask.sum()/len(y)*abs(conf-acc)
            bins.append({"bin":f"{lo:.1f}-{hi:.1f}","n":int(mask.sum()),"confidence":conf,"observed":acc})
    return {"ece":ece,"bins":bins}

def _metrics(rows,probs,theta=.5):
    names=("DOWN","SIDEWAYS","UP"); label_key=f"label_{theta}"; y=np.asarray([names.index(r[label_key]) for r in rows]); pred=probs.argmax(axis=1)
    cm=confusion_matrix(y,pred,labels=[0,1,2]);pr,re,fs,_=precision_recall_fscore_support(y,pred,labels=[0,1,2],zero_division=0)
    briers=[];eces={}
    for j,name in enumerate(names):
        actual=(y==j).astype(float); briers.append(float(np.mean((probs[:,j]-actual)**2))); eces[name]=_ece(y,probs,j)
    out={"n":len(rows),"accuracy_pct":100*accuracy_score(y,pred),"balanced_accuracy_pct":100*balanced_accuracy_score(y,pred),
      "macro_f1_pct":100*f1_score(y,pred,average="macro",zero_division=0),"class_metrics":{},
      "confusion_matrix":{"labels":names,"counts":cm.tolist()},"multiclass_log_loss":log_loss(y,probs,labels=[0,1,2]),
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

def _backtest_model(rows,train_rows,kind,horizon,period):
    lo,hi={"development":(SAMPLE_START,TRAIN_END),"validation":(TRAIN_END,VALID_END),"diagnostic":(VALID_END,HOLDOUT_END)}[period]
    eval_rows=[r for r in rows if r["horizon"]==horizon and lo<=r["entry_date"]<hi and r["label_available_date"]<hi]
    if not eval_rows:return {"n":0,"metrics":None,"buy_policy":None}
    eval_rows.sort(key=lambda r:(r["signal_date"],r["ticker"]))
    all_train=sorted([r for r in train_rows if r["horizon"]==horizon],key=lambda r:(r["label_available_date"],r["ticker"]))
    classes=("DOWN","SIDEWAYS","UP"); probs=np.full((len(eval_rows),3),1/3,dtype=float)
    target_probs=np.full(len(eval_rows),.5,dtype=float); scored=[]
    valid_years=set()
    for year in sorted({int(r["signal_date"][:4]) for r in eval_rows}):
        snapshot_year=year-(year%2)
        anchor=f"{snapshot_year}-01-01"; idx=[i for i,r in enumerate(eval_rows) if int(r["signal_date"][:4])//2==snapshot_year//2]
        past=[r for r in all_train if r["label_available_date"]<anchor and r["signal_date"]<anchor]
        # The primary classifier is fixed at net ±0.5%. Sensitivity labels
        # reuse this one set of causal probabilities below; no repeated fit.
        label_key="label_0.5"
        if len(past)<MIN_TRAIN or len({r[label_key] for r in past})<3:continue
        active=_active_features(past)
        if not active:continue
        model=_model(kind);model.fit(_features(past,active),np.asarray([classes.index(r[label_key]) for r in past]))
        raw=model.predict_proba(_features([eval_rows[i] for i in idx],active)); cls=list(model.classes_)
        p=np.zeros((len(idx),3));
        for j,c in enumerate(cls):p[:,int(c)]=raw[:,j]
        probs[idx]=p;valid_years.add(str(year))
        yhit=np.asarray([int(r.get("target_touch_theta_0.5")=="TARGET_FIRST") for r in past])
        if np.unique(yhit).size==2:
            hit_model=_model(kind);hit_model.fit(_features(past,active),yhit)
            hit_raw=hit_model.predict_proba(_features([eval_rows[i] for i in idx],active))
            hit_classes=list(hit_model.classes_)
            if 1 in hit_classes:target_probs[idx]=hit_raw[:,hit_classes.index(1)]
    valid=np.asarray([r["signal_date"][:4] in valid_years for r in eval_rows])
    eval_rows=[r for r,v in zip(eval_rows,valid) if v];probs=probs[valid];target_probs=target_probs[valid]
    if not eval_rows:return {"n":0,"metrics":None,"buy_policy":None}
    labels=np.asarray([classes.index(r["label_0.5"]) for r in eval_rows])
    valid=np.asarray([r["label_available_date"]< (TRAIN_END if period=="development" else VALID_END if period=="validation" else HOLDOUT_END) for r in eval_rows])
    eval_rows=[r for r,v in zip(eval_rows,valid) if v];probs=probs[valid]; labels=labels[valid]
    if not eval_rows:return {"n":0,"metrics":None,"buy_policy":None}
    target_actual=np.asarray([int(r.get("target_touch_theta_0.5")=="TARGET_FIRST") for r in eval_rows],dtype=float)
    target_bins=[];target_ece=0.
    for b in range(10):
        mask=(target_probs>=b/10)&((target_probs<(b+1)/10) if b<9 else (target_probs<=1))
        if mask.any():
            conf=float(target_probs[mask].mean());obs=float(target_actual[mask].mean())
            target_ece+=mask.sum()/len(eval_rows)*abs(conf-obs)
            target_bins.append({"bin":f"{b/10:.1f}-{(b+1)/10:.1f}","n":int(mask.sum()),"mean_probability":conf,"observed_target_first_pct":100*obs})
    target_hit_report={"target":"gross +0.5% touch before gross -0.5% stop; same-bar tie stop-first",
      "n":len(eval_rows),"observed_target_first_pct":100*float(target_actual.mean()),
      "mean_predicted_probability_pct":100*float(target_probs.mean()),
      "brier":float(np.mean((target_probs-target_actual)**2)),"ece":target_ece,"reliability_bins":target_bins}
    up=probs[:,2]; down=probs[:,0]; side=probs[:,1]
    theta_results={}
    for theta in THETAS:
        m=_metrics(eval_rows,probs,theta)
        m["target_hit_probability"]=target_hit_report
        policy_candidates={}
        for cut in (.45,.50,.55):
            buy=(up>=cut)&(down<.30)&(side<.50)
            selected=[r for r,b in zip(eval_rows,buy) if b]
            policy_candidates[f"p_up_ge_{cut:.2f}"]=_trade_perf(selected,eval_rows,probs,theta)
        theta_results[str(theta)]={"metrics":m,"buy_policy_candidates":policy_candidates,
                                   "buy_policy":policy_candidates["p_up_ge_0.50"]}
    primary=theta_results["0.5"]
    return {"n":len(eval_rows),"metrics":primary["metrics"],"theta_results":theta_results,
            "buy_policy":primary["buy_policy"],"buy_policy_candidates":primary["buy_policy_candidates"]}

def run_expanded_direction_audit():
    from concurrent.futures import ThreadPoolExecutor, as_completed
    universe=_ticker_set(); audit=_audit_symbols(universe); rows=[]; loaded=[]; failures=[]; special_eval={}
    sources=universe.pop("universe_sources",{}); universe_errors=universe.pop("universe_errors",{})
    jobs=[(c,item) for c,items in universe.items() for item in items]
    with ThreadPoolExecutor(max_workers=12) as pool:
      futs={pool.submit(get_historical_prices,item[0],item[2],start_date=SAMPLE_START,min_bars=250):(cohort,item)
            for cohort,item in jobs}
      for fut in as_completed(futs):
        cohort,(ticker,name,market,sector)=futs[fut]
        try:
            frame,source=fut.result()
            first=pd.Timestamp(frame.index.min()).date().isoformat()
            if pd.Timestamp(frame.index.max()).date().isoformat()<HOLDOUT_END:raise ValueError("history ends before fixed HOLDOUT_END")
            made=_make_rows(frame,ticker,market,cohort);rows.extend(made)
            loaded.append({"ticker":ticker,"cohort":cohort,"first_date":first,"source":source,"rows":len(made)})
        except Exception as exc:failures.append({"ticker":ticker,"cohort":cohort,"reason":str(exc)[:220]})
    rows=_rank_features(rows)
    audit_tickers=set().union(*audit.values())
    bycohort={c:[r for r in rows if r["cohort"]==c] for c in universe}
    # Keep the mandated leveraged ETF in the 63-instrument evaluation. It is
    # never mixed into the ordinary-US cross-section used to train the model.
    for ticker,name,market,sector in MANDATORY_US:
        try:
            frame,source=get_historical_prices(ticker,market,start_date=SAMPLE_START,min_bars=250)
            special_eval.setdefault("US",[]).extend(_make_rows(frame,ticker,market,"US"))
            loaded.append({"ticker":ticker,"cohort":"US_eval_only","first_date":str(frame.index.min().date()),"source":source,"rows":len(special_eval["US"])})
        except Exception as exc:failures.append({"ticker":ticker,"cohort":"US_eval_only","reason":str(exc)[:220]})
    for ticker,name,market,sector in MANDATORY_KR:
        try:
            frame,source=get_historical_prices(ticker,market,start_date=SAMPLE_START,min_bars=250)
            special_eval.setdefault("TIGER_ETF",[]).extend(_make_rows(frame,ticker,market,"TIGER_ETF"))
            loaded.append({"ticker":ticker,"cohort":"TIGER_ETF_eval_only","first_date":str(frame.index.min().date()),"source":source,"rows":len(special_eval["TIGER_ETF"])})
        except Exception as exc:failures.append({"ticker":ticker,"cohort":"TIGER_ETF_eval_only","reason":str(exc)[:220]})
    # ETF ranks are only within the two audited ETFs on the same date.
    for c in ("US","TIGER_ETF"):
        special_eval[c]=_rank_features(special_eval.get(c,[]))
    equity_loaded=sum(1 for row in loaded if row.get("cohort") in universe)
    out={"universe_expansion":{"requested":sum(map(len,universe.values())),"loaded":equity_loaded,
      "evaluation_only_loaded":len(loaded)-equity_loaded,
      "failures":failures,"loaded_tickers":loaded,"training_pool_candidate_rows":{c:len(v) for c,v in bycohort.items()},
      "universe_sources":sources,"universe_lookup_errors":universe_errors,
      "audit63_scored_symbols":{c:len(set(r["ticker"] for r in v)&audit.get(c,set())) for c,v in bycohort.items()},
      "cross_sectional_scope":"same signal date and same cohort only; missing symbols are excluded from that date rank",
      "periods":{"development_end":TRAIN_END,"validation_end":VALID_END,"diagnostic_end":HOLDOUT_END}},
      "label_variants":{"entry":"t+1 open","exit":"t+h close","horizons":list(HORIZONS),
      "cost_pct":COST_PCT,"direction_thresholds_pct":list(THETAS),"sensitivity_selected_horizon":None},
      "direction3":{},"selection_rule":"Development only: choose horizon/model by daily grouped Spearman of P(UP) vs net return, then top 15% mean net subject to >=100 selected development samples. Fixed buy rule P(UP)>=0.50, P(DOWN)<0.30, P(SIDEWAYS)<0.50.",
      "feature_coverage":{"feature_names":list(FEATURES),"fixed_centers":CENTERS,"fixed_scales":SCALES,"feature_count":len(FEATURES),"cross_section_count_key":"xs_count","constant_column_policy":"recomputed from each past-only training snapshot and excluded when std is zero"}}
    for cohort,pool in bycohort.items():
        eval_rows=[r for r in pool if r["ticker"] in audit.get(cohort,set())]
        eval_rows.extend(special_eval.get(cohort,[]))
        pool63=[r for r in eval_rows]
        choices=[];development_cache={}
        for horizon in HORIZONS:
          for model in ("multinomial_logistic","random_forest"):
            dev=_backtest_model(eval_rows,pool63,model,horizon,"development")
            development_cache[(model,horizon)]=dev
            met=dev.get("metrics") or {}; ic=met.get("datewise_spearman_mean"); top=met.get("top15_mean_net_pct")
            score=(float(ic)+.01*float(top)) if ic is not None and top is not None and met.get("top15_n",0)>=100 else -1e9
            choices.append((score,model,horizon))
        chosen=max(choices,default=(None,None,None))
        cohort_report={"selected_model_development_only":chosen[1],"selected_horizon_development_only":chosen[2],
          "selected_model_horizon_development_score":chosen[0],"universe_comparison":{},"periods":{}}
        for fit_name,training in (("63_audit_symbols",pool63),("available_full_universe",pool)):
          comparisons={};selected_cuts={};selected_oos={}
          raw_by_horizon={}
          for horizon in HORIZONS:
            raw_by_horizon[horizon]={}
            for period in ("development","validation","diagnostic"):
              if fit_name=="63_audit_symbols" and period=="development" and chosen[1] is not None:
                raw=development_cache.get((chosen[1],horizon))
              else:
                raw=_backtest_model(eval_rows,training,chosen[1],horizon,period) if chosen[1] else {"n":0}
              raw_by_horizon[horizon][period]=raw or {"n":0}
          for theta in THETAS:
            for horizon in HORIZONS:
              period_rows={}
              for period,raw in raw_by_horizon[horizon].items():
                measured=(raw.get("theta_results") or {}).get(str(theta))
                period_rows[period]=({"n":raw.get("n",0),**measured} if measured else
                    {"n":0,"metrics":None,"buy_policy":None,"buy_policy_candidates":{}})
              dev=period_rows["development"];dev_metrics=dev.get("metrics") or {}
              dev_policies=dev.get("buy_policy_candidates",{})
              dev_ic=dev_metrics.get("datewise_spearman_mean")
              eligible=[(v.get("mean_net_pct"),key) for key,v in dev_policies.items()
                        if v.get("n",0)>=100 and v.get("mean_net_pct") is not None
                        and v.get("mean_net_pct")>0 and dev_ic is not None and dev_ic>0]
              selected_cut=max(eligible,default=(None,None))[1]
              key=f"theta_{theta}_horizon_{horizon}"
              selected_cuts[key]=selected_cut
              if selected_cut:
                selected_oos[key]={"selection":"development only","threshold":selected_cut,
                  "development_datewise_ic":dev_ic,"periods":{p:(v.get("buy_policy_candidates",{}).get(selected_cut)) for p,v in period_rows.items()}}
              else:
                selected_oos[key]={"selection":"NO_DEVELOPMENT_EDGE","threshold":None,
                  "periods":{p:{"n":0,"win_rate_pct":None,"mean_net_pct":None,"coverage_pct":0.0} for p in period_rows}}
              comparisons[key]=period_rows
          cohort_report["universe_comparison"][fit_name]=comparisons
          cohort_report.setdefault("selected_buy_cut_by_development",{})[fit_name]=selected_cuts
          cohort_report.setdefault("selected_policy_results_by_development",{})[fit_name]=selected_oos
        cohort_report["policy_status"]=("SELECTED_ON_DEVELOPMENT_ONLY" if any(v for d in cohort_report.get("selected_buy_cut_by_development",{}).values() for v in d.values())
          else "NO_DEVELOPMENT_EDGE")
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
    # Preserve readable, auditable source/data failures separately.
    Path("data").mkdir(exist_ok=True)
    return out

if __name__=="__main__":
    result=run_expanded_direction_audit()
    Path("data/expanded-direction-latest.json").write_text(json.dumps(result,ensure_ascii=False,indent=2,default=str),encoding="utf-8")
    print(json.dumps({"requested":result["universe_expansion"]["requested"],"loaded":result["universe_expansion"]["loaded"],"failures":len(result["universe_expansion"]["failures"])},ensure_ascii=False),flush=True)
