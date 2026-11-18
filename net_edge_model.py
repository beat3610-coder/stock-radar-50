"""Net-return-first challenger: chronological model, abstention and loss attribution.

Only observable entry-time features are used. Model selection happens on
chronological DEVELOPMENT folds; neither validation nor diagnostic outcomes
select an algorithm or threshold. Historical diagnostic is not future evidence.
"""
from collections import Counter
import numpy as np
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor
from feature_selector60 import _clean
from profit_selector51 import TRAIN, VALID, END, stats

POLICIES={
 "positive_net_50":(.0,.50,1.2),
 "positive_net_55":(.0,.55,1.2),
 "edge035_50":(.35,.50,1.2),
 "edge035_55":(.35,.55,1.5),
 "edge070_55":(.70,.55,1.5),
}
MIN_FIT=60
MIN_FOLD=15

def _fit(rows,kind):
    if len(rows)<MIN_FIT or len({row[2] for row in rows})<2:return None
    x=np.stack([z[1] for z in rows])
    y=np.asarray([z[0]["net_pct"] for z in rows])
    if kind=="ridge":
        ret=make_pipeline(StandardScaler(),Ridge(alpha=200.0))
    else:
        ret=RandomForestRegressor(n_estimators=80,max_depth=3,
                  min_samples_leaf=20,random_state=7,n_jobs=1)
    ret.fit(x,np.clip(y,-12,12))
    prob=make_pipeline(StandardScaler(),LogisticRegression(C=0.2,max_iter=300))
    prob.fit(x,np.array([z[2] for z in rows]))
    return ret,prob

def _score(model,rows,policy):
    if not rows:return []
    x=np.stack([z[1] for z in rows])
    e=model[0].predict(x)
    p=model[1].predict_proba(x)[:,1]
    edge,prob,rr=POLICIES[policy]
    return [row[0] for row,net,win in zip(rows,e,p)
            if net>=edge and win>=prob and row[0].get("entry_net_rr",0)>=rr]

def _window(rows,begin,end):
    return [z for z in rows if begin<=z[0]["signal_date"]<end and z[0]["exit_date"]<end]

def _losses(rows):
    # Outcome-side attribution only, never fed back into the fit of this run.
    losses=[r for r in rows if r["net_pct"]<=0]
    return {"losses":len(losses),"total":len(rows),
        "loss_exit_reasons":dict(Counter(str(r.get("exit_reason","unknown")) for r in losses)),
        "average_loss_net_pct":round(sum(r["net_pct"] for r in losses)/len(losses),4) if losses else None}

def audit(trades,market=None):
    data=_clean(trades,market)
    dev=[x for x in data if x[0]["signal_date"]<TRAIN and x[0]["exit_date"]<TRAIN]
    val=_window(data,TRAIN,VALID)
    diag=_window(data,VALID,END)
    result={"market":market,"target_win_pct":60,"research_only":True,
            "live_approved":False,"status":"INSUFFICIENT_DEVELOPMENT",
            "selected_policy":None,"baseline":{"train":stats([z[0] for z in dev]),
              "validation":stats([z[0] for z in val]),
              "diagnostic":stats([z[0] for z in diag])},
            "diagnostic_loss_attribution":_losses([z[0] for z in diag])}
    dates=sorted({z[0]["signal_date"] for z in dev})
    if len(dates)<12 or len(dev)<90:return result
    dates_cut=[dates[int(len(dates)*f)] for f in (.45,.70)]
    fold_data=[]
    for i,cut in enumerate(dates_cut):
        end=dates_cut[i+1] if i+1<len(dates_cut) else TRAIN
        fit=[z for z in dev if z[0]["exit_date"]<cut]
        test=_window(dev,cut,end)
        if len(fit)<MIN_FIT or len(test)<MIN_FOLD:
            result["status"]="INSUFFICIENT_TIME_FOLDS"
            return result
        fold_data.append((fit,test))
    rankings=[]
    for kind in ("ridge","forest"):
        fits=[(_fit(training,kind),testing) for training,testing in fold_data]
        if any(model is None for model,_ in fits):continue
        for name in POLICIES:
            selected=[r for model,test in fits for r in _score(model,test,name)]
            m=stats(selected)
            if m["n"]<25 or m["mean_net_pct"]<=0:continue
            # Conservative utility: net expectancy and minimum-volume
            # preference, not artificially maximized win percentage.
            score=m["mean_net_pct"]-1.0/np.sqrt(m["n"])
            rankings.append((score,m["n"],kind,name,m))
    if not rankings:
        result["status"]="NO_POSITIVE_DEVELOPMENT_POLICY"
        return result
    rankings.sort(key=lambda z:(-z[0],-z[1],z[2],z[3]))
    _,_,kind,policy,fold=rankings[0]
    model=_fit(dev,kind)
    if model is None:return result
    vtr=_score(model,val,policy)
    dtr=_score(model,diag,policy)
    v=stats(vtr);d=stats(dtr)
    passed=(v["n"]>=30 and v["win_pct"]>=60 and v["mean_net_pct"]>0
            and (v["profit_factor"] is None and v["win_pct"]==100
                 or v["profit_factor"] is not None and v["profit_factor"]>1))
    result.update({"status":"VALIDATION_PASS_RESEARCH_ONLY" if passed else "VALIDATION_REJECTED",
        "selected_policy":policy,"selected_model":kind,
        "development_walkforward":fold,
        "validation":v,"diagnostic":d,"diagnostic_selected_loss_attribution":_losses(dtr),
        "validation_pass":passed,"diagnostic_coverage_pct":round(100*len(dtr)/len(diag),3) if diag else None})
    return result
