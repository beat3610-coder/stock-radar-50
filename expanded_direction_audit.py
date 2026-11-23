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
from scipy.optimize import minimize_scalar
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
    confusion_matrix, f1_score, log_loss, precision_recall_fscore_support,
    roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from providers import get_historical_prices
from universe import get_universe, MANDATORY_KR, MANDATORY_US, AUDIT_63_BENCHMARK
from engine import compute

TRAIN_END="2025-11-01"; VALID_END="2026-04-01"; HOLDOUT_END="2026-11-01"
SAMPLE_START="2016-01-01"; COST_PCT=.35; THETAS=(.3,.5,1.0)
HORIZONS=(5,10,20); MIN_TRAIN=500; REFIT_MONTH=1; SIGNAL_STEP=5
SIDEWAYS_TARGET_SHARE=.30
SPREAD_CUTS=(.05,.10,.15); SIDEWAYS_MAX=(.25,.35,.45); EV_FLOORS=(0.0,.10,.20)
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

def _active_features(rows):
    return tuple(k for k in FEATURES if np.std([float(r.get(k,CENTERS[k])) for r in rows])>1e-12)

def _features(rows,active=None):
    active=FEATURES if active is None else active
    return np.asarray([[np.clip((float(r.get(k,CENTERS[k]))-CENTERS[k])/SCALES[k],-5,5) for k in active]
                       for r in rows],dtype=np.float32)

def _model(kind):
    if kind=="multinomial_logistic":
        return make_pipeline(SimpleImputer(strategy="constant",fill_value=0),StandardScaler(),
             LogisticRegression(C=.15,max_iter=300,class_weight=None,random_state=3610))
    return RandomForestClassifier(n_estimators=80,max_depth=5,min_samples_leaf=20,
            class_weight=None,random_state=3610,n_jobs=1)

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
    fit=minimize_scalar(objective,bounds=(-1.5,1.5),method="bounded",options={"xatol":1e-5})
    return {"temperature":float(np.exp(fit.x)),"fit_n":len(y),"status":"DEVELOPMENT_OOF_TEMPERATURE_SCALING",
            "development_oof_logloss_before":objective(0.),"development_oof_logloss_after":float(fit.fun)}

def _choose_development_k(rows):
    """Compatibility alias for callers/tests; returns frozen cohort settings."""
    return _freeze_volatility_labels(rows)

def _ece(y,p,klass):
    confidence=p[:,klass]; actual=(y==klass).astype(float); ece=0.; bins=[]
    for b in range(10):
        lo=b/10; hi=(b+1)/10; mask=(confidence>=lo)&((confidence<hi) if b<9 else (confidence<=hi))
        if mask.any():
            conf=float(confidence[mask].mean()); acc=float(actual[mask].mean()); ece+=mask.sum()/len(y)*abs(conf-acc)
            bins.append({"bin":f"{lo:.1f}-{hi:.1f}","n":int(mask.sum()),"confidence":conf,"observed":acc})
    return {"ece":ece,"bins":bins}

def _metrics(rows,probs,theta=.5,label_key=None):
    names=("DOWN","SIDEWAYS","UP"); label_key=label_key or f"label_{theta}"; y=np.asarray([names.index(r[label_key]) for r in rows]); pred=probs.argmax(axis=1)
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

def _backtest_model(rows,train_rows,kind,horizon,period,temperature_info=None,payoffs=None):
    lo,hi={"development":(SAMPLE_START,TRAIN_END),"validation":(TRAIN_END,VALID_END),"diagnostic":(VALID_END,HOLDOUT_END)}[period]
    eval_rows=[r for r in rows if r["horizon"]==horizon and lo<=r["entry_date"]<hi and r["label_available_date"]<hi]
    if not eval_rows:return {"n":0,"metrics":None,"buy_policy":None,"temperature":temperature_info}
    eval_rows.sort(key=lambda r:(r["signal_date"],r["ticker"]))
    all_train=sorted([r for r in train_rows if r["horizon"]==horizon and "label_vol" in r],key=lambda r:(r["label_available_date"],r["ticker"]))
    classes=("DOWN","SIDEWAYS","UP"); probs=np.full((len(eval_rows),3),1/3,dtype=float)
    target_probs=np.full(len(eval_rows),.5,dtype=float);valid_years=set()
    for year in sorted({int(r["signal_date"][:4]) for r in eval_rows}):
        snapshot_year=year-(year%2); anchor=f"{snapshot_year}-01-01"
        idx=[i for i,r in enumerate(eval_rows) if int(r["signal_date"][:4])//2==snapshot_year//2]
        past=[r for r in all_train if r["label_available_date"]<anchor and r["signal_date"]<anchor]
        if len(past)<MIN_TRAIN or len({r["label_vol"] for r in past})<3:continue
        active=_active_features(past)
        if not active:continue
        model=_model(kind);model.fit(_features(past,active),np.asarray([classes.index(r["label_vol"]) for r in past]))
        raw=model.predict_proba(_features([eval_rows[i] for i in idx],active)); cls=list(model.classes_)
        p=np.zeros((len(idx),3))
        for j,c in enumerate(cls):p[:,int(c)]=raw[:,j]
        probs[idx]=p;valid_years.add(str(year))
        yhit=np.asarray([int(r.get("target_touch_theta_0.5")=="TARGET_FIRST") for r in past])
        if np.unique(yhit).size==2:
            hit_model=_model(kind);hit_model.fit(_features(past,active),yhit)
            hit_raw=hit_model.predict_proba(_features([eval_rows[i] for i in idx],active));hit_classes=list(hit_model.classes_)
            if 1 in hit_classes:target_probs[idx]=hit_raw[:,hit_classes.index(1)]
    valid=np.asarray([r["signal_date"][:4] in valid_years for r in eval_rows])
    eval_rows=[r for r,v in zip(eval_rows,valid) if v];probs=probs[valid];target_probs=target_probs[valid]
    if not eval_rows:return {"n":0,"metrics":None,"buy_policy":None,"temperature":temperature_info}
    valid=np.asarray([r["label_available_date"]<(TRAIN_END if period=="development" else VALID_END if period=="validation" else HOLDOUT_END) for r in eval_rows])
    eval_rows=[r for r,v in zip(eval_rows,valid) if v];probs=probs[valid];target_probs=target_probs[valid]
    if not eval_rows:return {"n":0,"metrics":None,"buy_policy":None,"temperature":temperature_info}
    if period=="development" and temperature_info is None:
        temperature_info=_fit_temperature(eval_rows,probs)
    temperature=float((temperature_info or {}).get("temperature",1.0))
    probs=_apply_temperature(probs,temperature)
    payoffs=payoffs or (_class_payoffs(eval_rows) if period=="development" else {"DOWN":-1.,"SIDEWAYS":0.,"UP":1.})
    theta_results={}
    for theta in THETAS:
        fixed=_metrics(eval_rows,probs,theta)
        theta_results[str(theta)]={"metrics":fixed}
    vol_metrics=_metrics(eval_rows,probs,label_key="label_vol")
    target_actual=np.asarray([int(r.get("target_touch_theta_0.5")=="TARGET_FIRST") for r in eval_rows],dtype=float)
    target_report={"n":len(eval_rows),"observed_target_first_pct":100*float(target_actual.mean()),
      "mean_predicted_probability_pct":100*float(target_probs.mean()),"brier":float(np.mean((target_probs-target_actual)**2))}
    vol_metrics["target_hit_probability"]=target_report
    policies=_policy_table(eval_rows,probs,payoffs)
    return {"n":len(eval_rows),"metrics":vol_metrics,"fixed_band_metrics":{k:v["metrics"] for k,v in theta_results.items()},
      "theta_results":theta_results,"buy_policy_candidates":policies,"buy_policy":None,
      "temperature":temperature_info,"class_payoffs_from_development":payoffs,
      "policy_selection_scope":"development only; thresholds are spread, sideways-probability, and net EV after existing costs",
      "probability_sample":{"p_up_p10_p50_p90":np.percentile(probs[:,2],[10,50,90]).tolist(),
        "p_down_p10_p50_p90":np.percentile(probs[:,0],[10,50,90]).tolist(),
        "p_sideways_p10_p50_p90":np.percentile(probs[:,1],[10,50,90]).tolist()}}

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
    for ticker,name,market in _mandatory_market_items(MANDATORY_US,"US"):
        try:
            frame,source=get_historical_prices(ticker,market,start_date=SAMPLE_START,min_bars=250)
            special_eval.setdefault("US",[]).extend(_make_rows(frame,ticker,market,"US"))
            loaded.append({"ticker":ticker,"cohort":"US_eval_only","first_date":str(frame.index.min().date()),"source":source,"rows":len(special_eval["US"])})
        except Exception as exc:failures.append({"ticker":ticker,"cohort":"US_eval_only","reason":str(exc)[:220]})
    for ticker,name,market in _mandatory_market_items(MANDATORY_KR,"KR"):
        try:
            frame,source=get_historical_prices(ticker,market,start_date=SAMPLE_START,min_bars=250)
            special_eval.setdefault("TIGER_ETF",[]).extend(_make_rows(frame,ticker,market,"TIGER_ETF"))
            loaded.append({"ticker":ticker,"cohort":"TIGER_ETF_eval_only","first_date":str(frame.index.min().date()),"source":source,"rows":len(special_eval["TIGER_ETF"])})
        except Exception as exc:failures.append({"ticker":ticker,"cohort":"TIGER_ETF_eval_only","reason":str(exc)[:220]})
    # ETF ranks are only within the two audited ETFs on the same date.
    for c in ("US","TIGER_ETF"):
        special_eval[c]=_rank_features(special_eval.get(c,[]))
    volatility_k=_freeze_volatility_labels(rows+special_eval.get("US",[])+special_eval.get("TIGER_ETF",[]))
    bycohort={c:[r for r in rows if r["cohort"]==c] for c in universe}
    equity_loaded=sum(1 for row in loaded if row.get("cohort") in universe)
    out={"universe_expansion":{"requested":sum(map(len,universe.values())),"loaded":equity_loaded,
      "evaluation_only_loaded":len(loaded)-equity_loaded,
      "failures":failures,"loaded_tickers":loaded,"training_pool_candidate_rows":{c:len(v) for c,v in bycohort.items()},
      "universe_sources":sources,"universe_lookup_errors":universe_errors,
      "audit63_scored_symbols":{c:len(set(r["ticker"] for r in v)&audit.get(c,set())) for c,v in bycohort.items()},
      "frozen_audit_benchmark":{c:list(v) for c,v in AUDIT_63_BENCHMARK.items()},
      "benchmark_missing_from_training_pool":{c:sorted(set(AUDIT_63_BENCHMARK.get(c,()))-set(x["ticker"] for x in bycohort.get(c,[]))) for c in bycohort},
      "cross_sectional_scope":"same signal date and same cohort only; missing symbols are excluded from that date rank",
      "periods":{"development_end":TRAIN_END,"validation_end":VALID_END,"diagnostic_end":HOLDOUT_END}},
      "label_variants":{"entry":"t+1 open","exit":"t+h close","horizons":list(HORIZONS),
      "cost_pct":COST_PCT,"legacy_fixed_thresholds_pct":list(THETAS),"sensitivity_selected_horizon":None,
      "volatility_band":{"formula":"k_cohort * signal_day_realized_vol20_pct * sqrt(h/20)","target_development_sideways_share_pct":100*SIDEWAYS_TARGET_SHARE,
        "cohort_parameters":volatility_k}},
      "direction3":{"label_definition":"net return after existing cost; volatility-scaled symmetric SIDEWAYS band; cohort k fitted using only outcomes available before TRAIN_END","cohort_k":volatility_k,"probability_calibration":{},"policy_results":{}},
      "selection_rule":"Development-only walk-forward: select one model/horizon by mean datewise Spearman(P(UP), realized net return), requiring positive IC and at least 100 top-15% development candidates. Calibrate multiclass probabilities with development OOF temperature scaling. Select spread + sideways-probability + net expected-value policy on development only; all selected policy OOS results are reported.",
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
        eligible_choices=[x for x in choices if x[0]>0 and (development_cache[(x[1],x[2])].get("metrics") or {}).get("datewise_spearman_mean",0)>0]
        chosen=max(eligible_choices,default=(-1e9,None,None))
        cohort_report={"selected_model_development_only":chosen[1],"selected_horizon_development_only":chosen[2],
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
        for fit_name,training in (("63_audit_symbols",pool63),("available_full_universe",pool)):
          comparisons={};chosen_policy=None;threshold_source=None
          if chosen[1] is not None:
            h=chosen[2]
            dev=(development_cache[(chosen[1],h)] if fit_name=="63_audit_symbols" else
                 _backtest_model(eval_rows,training,chosen[1],h,"development"))
            dev_payoffs=dev.get("class_payoffs_from_development") or _class_payoffs([r for r in training if r["horizon"]==h and r["label_available_date"]<TRAIN_END])
            dev_grid=dev.get("buy_policy_candidates",{})
            eligible=[(v.get("mean_net_pct"),key) for key,v in dev_grid.items()
                      if v.get("n",0)>=MIN_TRAIN and v.get("mean_net_pct") is not None and v.get("mean_net_pct")>0]
            chosen_policy=max(eligible,default=(None,None),key=lambda x:(x[0],x[1]))[1]
            threshold_source={"policy":chosen_policy,"calibration":dev.get("temperature"),"class_payoffs":dev_payoffs,
              "selection_rule":"highest development mean net among predeclared spread/side/netEV grid with at least MIN_TRAIN accepted development trades; policy selection does not use validation/diagnostic"}
            periods={"development":dev}
            for period in ("validation","diagnostic"):
                periods[period]=_backtest_model(eval_rows,training,chosen[1],h,period,
                  temperature_info=dev.get("temperature"),payoffs=dev_payoffs)
            for period,raw in periods.items():
                met=raw.get("metrics");grid=raw.get("buy_policy_candidates",{})
                comparisons[period]={"n":raw.get("n",0),"metrics":met,"fixed_band_metrics":raw.get("fixed_band_metrics"),
                  "selected_policy":chosen_policy,"selected_policy_metrics":grid.get(chosen_policy) if chosen_policy else None,
                  "all_development_grid_policy_results":grid if period=="development" else None,
                  "probability_sample":raw.get("probability_sample")}
            if fit_name=="63_audit_symbols":
                cohort_report["development_policy_grid"]=dev_grid
                cohort_report["selected_policy_thresholds"]=threshold_source
            else:
                cohort_report["full_universe_development_policy_grid"]=dev_grid
          else:
            comparisons={p:{"n":0,"metrics":None,"selected_policy":None,"selected_policy_metrics":None} for p in ("development","validation","diagnostic")}
          cohort_report["universe_comparison"][fit_name]=comparisons
          if chosen[1] is not None:
            cohort_report.setdefault("calibration_by_universe",{})[fit_name]=threshold_source["calibration"]
          cohort_report.setdefault("policy_selection_status_by_universe",{})[fit_name]=("DEVELOPMENT_POLICY_SELECTED" if chosen_policy else "NO_DEVELOPMENT_EDGE")
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
    out["direction3"]["calibration_scope"]="development OOF probabilities only; one scalar temperature per selected model/horizon/training universe; applied unchanged to validation and diagnostic"
    out["direction3"]["probability_calibration"]={cohort:value.get("calibration_by_universe",{})
        for cohort,value in out["direction3"].items() if isinstance(value,dict) and "universe_comparison" in value}
    out["direction3"]["policy_results"]={cohort:{universe_name:periods for universe_name,periods in value.get("universe_comparison",{}).items()}
        for cohort,value in out["direction3"].items() if isinstance(value,dict) and "universe_comparison" in value}
    out["direction3"]["label_definition_comparison"]={cohort:{"volatility_band":v,
      "fixed_pct_band_development_class_counts":{
        str(theta):{name:sum(r.get(f"label_{theta}")==name for r in bycohort.get(cohort,[]) if r["label_available_date"]<TRAIN_END)
                    for name in ("DOWN","SIDEWAYS","UP")} for theta in THETAS}}
      for cohort,v in volatility_k.items()}
    # Preserve readable, auditable source/data failures separately.
    Path("data").mkdir(exist_ok=True)
    return out

if __name__=="__main__":
    result=run_expanded_direction_audit()
    Path("data/expanded-direction-latest.json").write_text(json.dumps(result,ensure_ascii=False,indent=2,default=str),encoding="utf-8")
    print(json.dumps({"requested":result["universe_expansion"]["requested"],"loaded":result["universe_expansion"]["loaded"],"failures":len(result["universe_expansion"]["failures"])},ensure_ascii=False),flush=True)
