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
        if len(batch)<3:
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

def _fit(train,kind):
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

def _causal_probabilities(events, historical, raw_probs, reference_prior,
                          window=1200, alpha=0.0, temperature=1.0):
    """Adapt to recent matured labels without inspecting the signal's future.

    A fixed model may be shifted toward the most recent *known* class mix,
    but the normalization reference is from the model's own training sample.
    """
    import numpy as np
    from collections import deque
    history=sorted(((r[0]["label_available_date"],y) for r,y in
                    ((row,label) for row,_,label in historical)))
    ordered=sorted(range(len(events)), key=lambda i:events[i][0]["signal_date"])
    mature=0
    last=deque(maxlen=window)
    result=np.zeros_like(raw_probs)
    for ix in ordered:
        day=events[ix][0]["signal_date"]
        while mature<len(history) and history[mature][0]<day:
            last.append(history[mature][1])
            mature+=1
        # New known outcomes in the evaluation window, always strictly matured.
        # They may affect only later predictions, not the model fit itself.
        counts=np.bincount(list(last),minlength=3).astype(float)+30*reference_prior
        recent=counts/sum(counts)
        ratio=np.power(np.maximum(recent,1e-6)/np.maximum(reference_prior,1e-6),alpha)
        q=np.power(np.maximum(raw_probs[ix],1e-7),1/temperature)*ratio
        result[ix]=q/q.sum()
    return result

def _causal_recalibration(events, history, raw_probs, reference_prior,
                          window=1200, alpha=0.0, temperature=1.0):
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
        if alpha and len(counts)>=60:
            recent=class_counts+20*reference_prior
            recent=recent/recent.sum()
            shift=np.power(np.maximum(recent,1e-8)/
                           np.maximum(reference_prior,1e-8),alpha)
        else:
            shift=np.ones(3)
        q=np.power(np.maximum(raw_probs[ix],1e-6),1/temperature)*shift
        result[ix]=q/q.sum()
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
    for kind in ("histgb","extratrees"):
        model=_fit(pretrain,kind)
        raw=_predict(model,validation)
        for window in (800,2500):
            for alpha in (0.0,0.5,1.0):
                for temperature in (0.75,1.0,1.25):
                    p=_causal_recalibration(validation,pretrain,raw,prior,
                                            window,alpha,temperature)
                    score=_stats(yv,p)
                    # Make class collapse explicit in selection.
                    active=sum(score["predicted"][name]>=20 for name in LABELS)
                    candidates.append({"model":kind,"window":window,
                                       "alpha":alpha,"temperature":temperature,
                                       "active_classes":active,**score})
    baseline_brier=_stats(yv,np.tile(prior,(len(yv),1)))["brier"]
    eligible=[r for r in candidates if r["active_classes"]==3]
    if not eligible:
        eligible=candidates
    chosen=min(eligible,key=lambda r:(r["brier"],
                       -r["macro_f1_pct"],-r["accuracy_pct"]))
    model=_fit(fulltrain,chosen["model"])
    refit_prior=np.bincount([y for _,_,y in fulltrain],minlength=3).astype(float)+1
    refit_prior/=refit_prior.sum()
    raw=_predict(model,diagnostic)
    p=_causal_recalibration(diagnostic,fulltrain,raw,refit_prior,
                            chosen["window"],chosen["alpha"],chosen["temperature"])
    yd=np.asarray([y for _,_,y in diagnostic],dtype=int)
    result=_stats(yd,p)
    return {"status":"RESEARCH_ONLY","train_mature":len(pretrain),
            "validation_count":len(validation),"refit_mature":len(fulltrain),
            "diagnostic_count":len(diagnostic),
            "chosen_on_validation":{"model":chosen["model"],
               "recent_mature_window":chosen["window"],
               "prior_adaptation_strength":chosen["alpha"],
               "temperature":chosen["temperature"],
               "validation_accuracy_pct":chosen["accuracy_pct"],
               "validation_macro_f1_pct":chosen["macro_f1_pct"],
               "validation_brier":chosen["brier"],
               "prior_only_brier":baseline_brier,
               "validation_predicted":chosen["predicted"]},
            "diagnostic":result,
            "diagnostic_collapse":any(v<20 for v in result["predicted"].values()),
            "notes":["Nonlinear model with causal rolling class prior",
                     "No signal-day label used in any prediction",
                     "Validation-only hyperparameter choice",
                     "Historical diagnostic previously inspected; not forward proof",
                     "Trade profitability not evaluated here"]}
