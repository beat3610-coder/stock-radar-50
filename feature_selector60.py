"""60% research: train-only feature model, fixed holdout gates, no look-ahead.

Unlike market-only regime counters, this learns nine features of each trade.
The algorithm/threshold are selected ONLY by internal chronological development
folds and never by external validation or already-inspected diagnostic outcomes.
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from profit_selector51 import TRAIN, VALID, END, stats

FEATURES=("signal_rsi","signal_adx","signal_atr_pct","signal_ma20_ma60_pct",
          "signal_volume_ratio","signal_5d_return_pct","entry_gap_pct",
          "entry_net_rr","entry_net_upside_pct")
THRESHOLDS=(0.45,0.50,0.55,0.60,0.65)

def _clean(trades,market):
    rows=[]
    for r in trades:
        if market is not None and r.get("market")!=market:continue
        sig=str(r.get("signal_date",""))[:10]
        end=str(r.get("exit_date",""))[:10]
        if len(sig)!=10 or len(end)!=10 or end<sig or sig>=END:continue
        try:
            features=np.asarray([float(r[k]) for k in FEATURES],float)
            net=float(r["net_pct"])
        except (KeyError,ValueError,TypeError,OverflowError):continue
        if not np.isfinite(features).all() or not np.isfinite(net):continue
        rows.append(({**r,"signal_date":sig,"exit_date":end,"net_pct":net},features,int(net>0)))
    return sorted(rows,key=lambda z:(z[0]["signal_date"],str(z[0].get("ticker",""))))

def _model(kind):
    if kind=="logistic":
        return make_pipeline(StandardScaler(),LogisticRegression(C=0.2,max_iter=300,random_state=17))
    if kind=="forest":
        return RandomForestClassifier(n_estimators=120,max_depth=4,min_samples_leaf=16,
            max_features=0.8,random_state=17,n_jobs=1)
    return HistGradientBoostingClassifier(max_iter=60,max_leaf_nodes=7,max_depth=2,
        min_samples_leaf=24,learning_rate=0.04,l2_regularization=12,random_state=17)

def _fit(train,kind):
    labels=[v[2] for v in train]
    if len(train)<60 or len(set(labels))<2:return None
    model=_model(kind)
    model.fit(np.stack([v[1] for v in train]),labels)
    return model

def _predict(model,rows,prior):
    if not rows:return np.array([],dtype=float)
    raw=model.predict_proba(np.stack([v[1] for v in rows]))[:,1]
    return .8*raw+.2*prior

def _score(rows,p,threshold):
    picked=[v[0] for v,prob in zip(rows,p) if prob>=threshold]
    result=stats(picked)
    result["coverage_pct"]=round(100*len(picked)/len(rows),3) if rows else None
    return result

def audit(trades,market=None):
    rows=_clean(trades,market)
    dev=[r for r in rows if r[0]["signal_date"]<TRAIN and r[0]["exit_date"]<TRAIN]
    validation=[r for r in rows if TRAIN<=r[0]["signal_date"] and r[0]["exit_date"]<VALID]
    diagnostic=[r for r in rows if VALID<=r[0]["signal_date"] and r[0]["exit_date"]<END]
    result={"market":market,"target_win_pct":60,"input_rows":len(rows),
        "eligible_for_live":False,"status":"INSUFFICIENT_DATA",
        "baseline":{"train":stats([x[0] for x in dev]),
          "validation":stats([x[0] for x in validation]),
          "diagnostic":stats([x[0] for x in diagnostic])},
        "selected":None}
    # Fixed prior model when nested folds cannot be built; no holdout tuning.
    def fixed_logistic(reason):
        frozen=_fit(dev,"logistic")
        if frozen is None:
            result["insufficient_reason"]=reason
            return result
        prior=np.mean([r[2] for r in dev])
        vm=_score(validation,_predict(frozen,validation,prior),0.50)
        dm=_score(diagnostic,_predict(frozen,diagnostic,prior),0.50)
        result.update({"status":"FIXED_BASELINE_RESEARCH_ONLY",
          "selected":{"model":"logistic_fixed_fallback","threshold":0.50,
            "internal_training":None,"validation":vm,
            "diagnostic_research_only":dm,"validation_pass":False},
          "insufficient_reason":reason})
        return result
    days=sorted(set(r[0]["signal_date"] for r in dev))
    if len(days)<18 or len(dev)<125:return fixed_logistic("too_few_distinct_training_dates_or_trades")
    cuts=[days[int(len(days)*k)] for k in (.40,.70)]
    folds=[]
    for i,cut in enumerate(cuts):
        until=cuts[i+1] if i+1<len(cuts) else TRAIN
        fit=[r for r in dev if r[0]["exit_date"]<cut]
        out=[r for r in dev if cut<=r[0]["signal_date"]<until]
        if len(fit)<60 or len(out)<15:return fixed_logistic("time_fold_insufficient_training_or_test_rows")
        folds.append((fit,out))
    candidates=[]
    for kind in ("logistic","forest","boost"):
        fold_preds=[]
        for fit,out in folds:
            model=_fit(fit,kind)
            if model is None:break
            fold_preds.append((out,_predict(model,out,np.mean([r[2] for r in fit]))))
        if len(fold_preds)!=len(folds):continue
        for threshold in THRESHOLDS:
            selected=[v[0] for out,p in fold_preds for v,prob in zip(out,p) if prob>=threshold]
            m=stats(selected)
            if m["n"]<25 or m["win_pct"]<55 or m["mean_net_pct"]<=0:continue
            # Penalize tiny high-hit-rate samples and overconfident cutoffs.
            utility=m["win_pct"]+min(m["mean_net_pct"],4)+min(m["n"],100)/40
            candidates.append((utility,kind,threshold,m))
    if not candidates:
        result["status"]="REJECTED_NO_POSITIVE_DEVELOPMENT_POLICY"
        return fixed_logistic("no_eligible_positive_internal_policy")
    candidates.sort(key=lambda r:(-r[0],r[1],r[2]))
    _,kind,threshold,internal=candidates[0]
    frozen=_fit(dev,kind)
    prior=np.mean([r[2] for r in dev])
    vm=_score(validation,_predict(frozen,validation,prior),threshold)
    dm=_score(diagnostic,_predict(frozen,diagnostic,prior),threshold)
    passed=vm["n"]>=30 and vm["win_pct"]>=60 and vm["mean_net_pct"]>0 and (
        vm["profit_factor"] is None and vm["win_pct"]==100 or
        vm["profit_factor"] is not None and vm["profit_factor"]>1)
    result.update({"status":"VALIDATION_PASS_RESEARCH_ONLY" if passed else "VALIDATION_REJECTED",
      "selected":{"model":kind,"threshold":threshold,"internal_training":internal,
        "validation":vm,"diagnostic_research_only":dm,"validation_pass":passed}})
    return result
