"""Nonlinear out-of-period challenger for 3-way stock direction.

Uses exactly the existing 10-session +/-3% labels. All cross-sectional
features are calculated from contemporaneously observable signals. Labels
used for each fit must mature strictly before its forecast window.
This remains retrospective *research*, with current-member survivorship bias.
"""
from collections import defaultdict
from direction3 import FEATURE_KEYS, LABELS

def market_features(rows):
    import numpy as np
    # Same-date market-wide momentum, breadth and volatility use no future bars.
    daily=defaultdict(list)
    for r in rows:
        daily[r["signal_date"]].append(r)
    context={}
    for day,batch in daily.items():
        momentum=np.array([float(r["signal_5d_return_pct"]) for r in batch],dtype=float)
        volatility=np.array([float(r["signal_atr_pct"]) for r in batch],dtype=float)
        spread=np.array([float(r["signal_ma20_ma60_pct"]) for r in batch],dtype=float)
        # Sparse fixed-watchlist studies may have nonaligned 2-session signal
        # sampling. Even one observed constituent is causal; mark low breadth
        # as a research limitation rather than dropping the entire session.
        if not batch:
            continue
        context[day]=(float(np.median(momentum)),
                      float(np.median(volatility)),
                      float(np.mean(momentum>0)),
                      float(np.median(spread)))
    return context

def vectors(rows, context):
    import numpy as np
    output=[]
    for row in rows:
        ctx=context.get(row["signal_date"])
        if ctx is None:
            continue
        try:
            raw=[float(row[k]) for k in FEATURE_KEYS]
            extra=[*ctx,raw[5]-ctx[0],raw[2]-ctx[1],
                   raw[3]-ctx[3]]
            vec=raw+extra
            if not all(np.isfinite(v) for v in vec):
                continue
            output.append((row,vec,LABELS.index(row["direction_label"])))
        except (ValueError,TypeError,KeyError):
            continue
    return output

def _stats(y,p):
    import numpy as np
    from sklearn.metrics import f1_score,accuracy_score
    if not len(y):
        return {"samples":0,"accuracy_pct":None,"macro_f1_pct":None,
                "brier":None,"predicted":{}}
    pred=np.argmax(p,axis=1)
    return {"samples":len(y),
            "accuracy_pct":round(100*accuracy_score(y,pred),2),
            "macro_f1_pct":round(100*f1_score(y,pred,labels=[0,1,2],
                                             average="macro",zero_division=0),2),
            "brier":round(float(np.mean(np.sum((p-np.eye(3)[y])**2,axis=1))),6),
            "predicted":{name:int((pred==i).sum()) for i,name in enumerate(LABELS)},
            "actual":{name:int((y==i).sum()) for i,name in enumerate(LABELS)},
            "direction_f1_pct":{name:round(100*f1_score(y==i,pred==i,zero_division=0),2)
                                for i,name in enumerate(LABELS)}}

def _fit(train,kind,half_life_days=None,as_of_date=None):
    import numpy as np
    from sklearn.ensemble import HistGradientBoostingClassifier, ExtraTreesClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.impute import SimpleImputer
    x=np.asarray([v for _,v,_ in train],dtype=float)
    y=np.asarray([label for _,_,label in train],dtype=int)
    if len(set(y))<3 or len(train)<240:
        raise ValueError("Insufficient mature training labels for all 3 classes")
    if kind=="histgb":
        model=HistGradientBoostingClassifier(max_iter=110,learning_rate=0.045,
            max_leaf_nodes=9,min_samples_leaf=75,l2_regularization=20.0,
            random_state=20261116)
    elif kind=="extratrees":
        model=ExtraTreesClassifier(n_estimators=100,max_depth=7,
            min_samples_leaf=45,max_features=0.65,n_jobs=-1,
            random_state=20261116)
    else:
        raise ValueError("invalid model")
    if half_life_days is not None:
        from datetime import date
        if not as_of_date or half_life_days<=0:
            raise ValueError("Weighted training requires a positive half-life and cutoff")
        cutoff=date.fromisoformat(as_of_date)
        ages=np.asarray([max(0,(cutoff-date.fromisoformat(r["signal_date"])).days)
                         for r,_,_ in train],dtype=float)
        sample_weight=np.maximum(0.05,np.exp2(-ages/half_life_days))
        model.fit(x,y,sample_weight=sample_weight)
    else:
        model.fit(x,y)
    return model

def _predict(model,rows):
    import numpy as np
    raw=model.predict_proba(np.asarray([v for _,v,_ in rows],dtype=float))
    # Classes are explicitly aligned to UP, DOWN, SIDEWAYS.
    p=np.zeros((len(rows),3),dtype=float)
    for idx,c in enumerate(model.classes_):
        p[:,int(c)]=raw[:,idx]
    return p

def _causal_recalibration(events, history, raw_probs, reference_prior,
                          window=1200, alpha=0.0, temperature=1.0,
                          prior_blend=0.0):
    """Online latest-observed class frequencies. An event never uses its own label."""
    import numpy as np
    from collections import deque
    observations=sorted(
        [(r[0]["label_available_date"],label) for r,label in
         [(item,item[2]) for item in history]]
        +[(r[0]["label_available_date"],label) for r,label in
          [(item,item[2]) for item in events]])
    sorted_idxs=sorted(range(len(events)),key=lambda k:events[k][0]["signal_date"])
    counts=deque()
    class_counts=np.zeros(3,dtype=float)
    result=np.zeros_like(raw_probs)
    j=0
    for ix in sorted_idxs:
        day=events[ix][0]["signal_date"]
        while j<len(observations) and observations[j][0]<day:
            cls=observations[j][1]
            counts.append(cls)
            class_counts[cls]+=1
            if len(counts)>window:
                class_counts[counts.popleft()]-=1
            j+=1
        if len(counts)>=60:
            recent=class_counts+20*reference_prior
            recent=recent/recent.sum()
            shift=np.power(np.maximum(recent,1e-8)/
                           np.maximum(reference_prior,1e-8),alpha)
        else:
            recent=reference_prior
            shift=np.ones(3)
        q=np.power(np.maximum(raw_probs[ix],1e-6),1/temperature)*shift
        q=q/q.sum()
        # Shrink overconfident tree scores toward a *matured* recent base rate.
        # Unlike relabeling predictions, this makes confidence reflect evidence.
        result[ix]=(1-prior_blend)*q+prior_blend*recent
    return result

def run_nonlinear(rows,train_end="2025-11-01",
                  validation_end="2026-04-01",eval_end="2026-11-01"):
    import numpy as np
    context=market_features(rows)
    valid=vectors(rows,context)
    pretrain=[r for r in valid if r[0]["label_available_date"]<train_end]
    validation=[r for r in valid if train_end<=r[0]["signal_date"]<validation_end
                and r[0]["label_available_date"]<validation_end]
    fulltrain=[r for r in valid if r[0]["label_available_date"]<validation_end]
    diagnostic=[r for r in valid if validation_end<=r[0]["signal_date"]<eval_end
                and r[0]["label_available_date"]<eval_end]
    if min(len(pretrain),len(validation),len(diagnostic))<100:
        return {"status":"INSUFFICIENT","train":len(pretrain),
                "validation":len(validation),"diagnostic":len(diagnostic)}
    yv=np.asarray([y for _,_,y in validation],dtype=int)
    prior=np.bincount([y for _,_,y in pretrain],minlength=3).astype(float)+1
    prior/=prior.sum()
    candidates=[]
    # Train both genuinely different nonlinear learners, then test a
    # *predeclared* equal-weight ensemble on the very same validation samples.
    validation_predictions={}
    for kind in ("histgb","extratrees"):
        validation_predictions[kind]=_predict(_fit(pretrain,kind),validation)
    validation_predictions["ensemble50"]=(
        validation_predictions["histgb"]+validation_predictions["extratrees"])/2
    for kind,raw in validation_predictions.items():
        for window in (800,2500):
            for alpha in (0.0,0.5,1.0):
                for temperature in (0.75,1.0,1.25):
                    for blend in (0.0,0.25,0.5,0.75):
                        p=_causal_recalibration(validation,pretrain,raw,prior,
                                    window,alpha,temperature,prior_blend=blend)
                        score=_stats(yv,p)
                        active=sum(score["predicted"][name]>=20 for name in LABELS)
                        candidates.append({"model":kind,"window":window,
                            "alpha":alpha,"temperature":temperature,
                            "prior_blend":blend,"active_classes":active,**score})
    baseline_brier=_stats(yv,np.tile(prior,(len(yv),1)))["brier"]
    eligible=[r for r in candidates if r["active_classes"]==3]
    if not eligible:
        eligible=candidates
    chosen=min(eligible,key=lambda r:(r["brier"],
                       -r["macro_f1_pct"],-r["accuracy_pct"]))
    if chosen["model"]=="ensemble50":
        raw=(_predict(_fit(fulltrain,"histgb"),diagnostic)+
             _predict(_fit(fulltrain,"extratrees"),diagnostic))/2
    else:
        raw=_predict(_fit(fulltrain,chosen["model"]),diagnostic)
    refit_prior=np.bincount([y for _,_,y in fulltrain],minlength=3).astype(float)+1
    refit_prior/=refit_prior.sum()
    p=_causal_recalibration(diagnostic,fulltrain,raw,refit_prior,
                            chosen["window"],chosen["alpha"],chosen["temperature"],
                            prior_blend=chosen["prior_blend"])
    yd=np.asarray([y for _,_,y in diagnostic],dtype=int)
    result=_stats(yd,p)
    # An uninformative class frequency is a necessary probability baseline.
    # Without beating it, higher quoted confidence is not justified.
    prior_only=np.tile(refit_prior,(len(yd),1))
    result["refit_prior_brier"]=_stats(yd,prior_only)["brier"]
    result["brier_skill_vs_prior"]=round(
        1-result["brier"]/result["refit_prior_brier"],5)
    # Conditional-accuracy numbers are NOT overall success rates; include coverage.
    predicted=np.argmax(p,axis=1)
    confidence=np.max(p,axis=1)
    result["confidence_bins"]={}
    for low,high in ((0.0,0.4),(0.4,0.5),(0.5,0.6),(0.6,0.7),(0.7,1.01)):
        keep=(confidence>=low)&(confidence<high)
        count=int(keep.sum())
        result["confidence_bins"][f"{low:.1f}-{high:.1f}"]={
            "count":count,"coverage_pct":round(100*count/len(yd),2),
            "accuracy_pct":round(100*float((predicted[keep]==yd[keep]).mean()),2)
                if count else None,
            "mean_predicted_confidence_pct":round(100*float(confidence[keep].mean()),2)
                if count else None}
    # Coverage is explicit: abstaining can improve conditional precision at
    # the cost of missing most opportunities. Threshold fixed on validation.
    val_raw=validation_predictions[chosen["model"]]
    val_p=_causal_recalibration(validation,pretrain,val_raw,prior,
        chosen["window"],chosen["alpha"],chosen["temperature"],
        prior_blend=chosen["prior_blend"])
    val_pred=np.argmax(val_p,axis=1)
    val_conf=np.max(val_p,axis=1)
    candidates_cutoffs=[]
    for cutoff in (0.4,0.45,0.5,0.55,0.6):
        mask=val_conf>=cutoff
        coverage=float(mask.mean())
        accuracy=float(np.mean(val_pred[mask]==yv[mask])) if mask.any() else 0.0
        if coverage>=0.10 and int(mask.sum())>=100:
            candidates_cutoffs.append((cutoff,coverage,accuracy))
    passing=[x for x in candidates_cutoffs if x[2]>=0.55]
    # Never advertise an unvalidated selective-success rate.
    chosen_cutoff=min(passing,key=lambda t:(-t[1],-t[2]))[0] if passing else None
    if chosen_cutoff is not None:
        keep=confidence>=chosen_cutoff
        selected=int(keep.sum())
        selective={"threshold":chosen_cutoff,"signals":selected,
                   "coverage_pct":round(100*selected/len(yd),2),
                   "conditional_accuracy_pct":round(100*float(
                       np.mean(predicted[keep]==yd[keep])),2) if selected else None,
                   "validation_target_met":True}
    else:
        selective={"threshold":None,"signals":0,"coverage_pct":0.0,
                   "conditional_accuracy_pct":None,"validation_target_met":False}
    result["selective_signal"]=selective
    ticker_metrics={}
    for ticker in sorted({item[0]["ticker"] for item in diagnostic}):
        indices=[i for i,item in enumerate(diagnostic) if item[0]["ticker"]==ticker]
        if indices:
            ticker_metrics[ticker]=_stats(yd[indices],p[indices])
    return {"status":"RESEARCH_ONLY","train_mature":len(pretrain),
            "validation_count":len(validation),"refit_mature":len(fulltrain),
            "diagnostic_count":len(diagnostic),
            "chosen_on_validation":{"model":chosen["model"],
               "recent_mature_window":chosen["window"],
               "prior_adaptation_strength":chosen["alpha"],
               "temperature":chosen["temperature"],
               "prior_blend":chosen["prior_blend"],
               "validation_accuracy_pct":chosen["accuracy_pct"],
               "validation_macro_f1_pct":chosen["macro_f1_pct"],
               "validation_brier":chosen["brier"],
               "prior_only_brier":baseline_brier,
               "validation_predicted":chosen["predicted"]},
            "diagnostic":result,"per_ticker_diagnostic":ticker_metrics,
            "diagnostic_collapse":any(v<20 for v in result["predicted"].values()),
            "notes":["Nonlinear model with causal rolling class prior",
                     "No signal-day label used in any prediction",
                     "Validation-only hyperparameter choice",
                     "Historical diagnostic previously inspected; not forward proof",
                     "Trade profitability not evaluated here"]}
