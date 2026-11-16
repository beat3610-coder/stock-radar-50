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

def run_nonlinear(rows,train_end="2025-11-01",
                  validation_end="2026-04-01",eval_end="2026-11-01"):
    import numpy as np
    from sklearn.metrics import log_loss
    context=market_features(rows)
    valid=vectors(rows,context)
    pretrain=[r for r in valid if r[0]["label_available_date"]<train_end]
    validation=[r for r in valid if train_end<=r[0]["signal_date"]<validation_end
                and r[0]["label_available_date"]<validation_end]
    # The second fit uses only outcomes already observable before the
    # diagnostic period; no late labels and no diagnostic labels.
    fulltrain=[r for r in valid if r[0]["label_available_date"]<validation_end]
    diagnostic=[r for r in valid if validation_end<=r[0]["signal_date"]<eval_end
                and r[0]["label_available_date"]<eval_end]
    if min(len(pretrain),len(validation),len(diagnostic))<100:
        return {"status":"INSUFFICIENT","train":len(pretrain),
                "validation":len(validation),"diagnostic":len(diagnostic)}
    chosen=None
    variants=[]
    yv=np.asarray([y for _,_,y in validation],dtype=int)
    for name in ("histgb","extratrees"):
        model=_fit(pretrain,name)
        valprobs=_predict(model,validation)
        prior=np.bincount([y for _,_,y in pretrain],minlength=3).astype(float)+1
        prior/=sum(prior)
        for shrink in (0.0,0.2,0.4,0.6):
            p=(1-shrink)*valprobs+shrink*prior
            metrics=_stats(yv,p)
            # Select on validation only; prefer baseline-beating calibration
            # and macro F1, then balanced accuracy.
            variants.append({"model":name,"shrink":shrink,**metrics})
    valid_prior=np.bincount([y for _,_,y in pretrain],minlength=3).astype(float)+1
    valid_prior/=sum(valid_prior)
    baseline_brier=_stats(yv,np.tile(valid_prior,(len(yv),1)))["brier"]
    # Selection by Brier, with minimum macro-F1. Do not optimize test period.
    chosen=min(variants,key=lambda r:(r["brier"],-r["macro_f1_pct"]))
    model=_fit(fulltrain,chosen["model"])
    prior=np.bincount([y for _,_,y in fulltrain],minlength=3).astype(float)+1
    prior/=sum(prior)
    pd=_predict(model,diagnostic)
    pd=(1-chosen["shrink"])*pd+chosen["shrink"]*prior
    yd=np.asarray([y for _,_,y in diagnostic],dtype=int)
    return {"status":"RESEARCH_ONLY",
            "train_mature":len(pretrain),"validation_count":len(validation),
            "refit_mature":len(fulltrain),"diagnostic_count":len(diagnostic),
            "chosen_on_validation":{"model":chosen["model"],"prior_shrink":chosen["shrink"],
                "validation_accuracy_pct":chosen["accuracy_pct"],
                "validation_macro_f1_pct":chosen["macro_f1_pct"],
                "validation_brier":chosen["brier"],
                "prior_only_brier":baseline_brier},
            "diagnostic":_stats(yd,pd),
            "notes":["Strictly mature labels before each training cutoff",
                     "Market-wide features from same-day signals only",
                     "Historical diagnostic period was previously inspected",
                     "This is not an executable trading strategy backtest"]}
