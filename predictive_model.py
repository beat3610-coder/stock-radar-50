"""Prequential, regularized win-probability model for historical candidate trades.

No external ML dependency. Each score is fitted solely to fully CLOSED prior
trades, with observable features recorded on the signal candle / entry open.
It must never fit or threshold against the evaluated candidate's outcome.
Research only; retrospective dates have already been inspected.
"""
import math
import numpy as np

MIN_COMPLETED=35
MAX_TRAIN=240
MIN_PREDICTED_WIN=0.55
MIN_UPLIFT=0.04
MIN_EXPECTED_NET_PCT=0.20

# Fixed scales are set from domain ranges, not from future data.
FEATURES=("signal_rsi","signal_adx","signal_atr_pct","signal_ma20_ma60_pct",
          "signal_volume_ratio","signal_5d_return_pct","entry_gap_pct")
CENTER=np.array([55.0,25.0,2.5,2.0,1.3,2.0,0.0],dtype=float)
SCALE=np.array([20.0,20.0,3.0,4.0,1.5,8.0,2.5],dtype=float)

def vector(trade):
    try:
        values=np.asarray([float(trade[key]) for key in FEATURES],dtype=float)
        if not np.isfinite(values).all():return None
        return np.clip((values-CENTER)/SCALE,-3,3)
    except (KeyError,TypeError,ValueError,OverflowError):
        return None

def _sigmoid(x):
    return 1/(1+np.exp(-np.clip(x,-35,35)))

def prepare_snapshot(history):
    """Fit once per signal date, using only trades exited before that date."""
    usable=[]
    for trade in history[-MAX_TRAIN:]:
        v=vector(trade)
        try: pnl=float(trade["net_pct"])
        except (ValueError,TypeError,KeyError):continue
        if v is not None and np.isfinite(pnl):usable.append((v,pnl))
    if len(usable)<MIN_COMPLETED:return None
    x=np.asarray([v for v,_ in usable])
    pnl=np.asarray([r for _,r in usable])
    y=(pnl>0).astype(float)
    n=len(y)
    if np.unique(y).size<2:return None
    X=np.column_stack([np.ones(n),x])
    beta=np.zeros(X.shape[1],dtype=float)
    penalty=np.diag([0.0]+[16.0]*x.shape[1])
    for _ in range(12):
        p=_sigmoid(X@beta)
        weights=np.maximum(p*(1-p),0.02)
        grad=X.T@(y-p)-penalty@beta
        hessian=X.T@(weights[:,None]*X)+penalty+np.eye(X.shape[1])*1e-7
        delta=np.linalg.solve(hessian,grad)
        beta+=np.clip(delta,-1.5,1.5)
        if np.max(np.abs(delta))<1e-5:break
    return {"n":n,"beta":beta,"base":float((y.sum()+2)/(n+4)),
            "win_gain":float(np.mean(pnl[y>0])),
            "loss_net":float(np.mean(pnl[y==0]))}

def predict_from_snapshot(model,candidate):
    """Fast feature-only prediction using a frozen prior-day model."""
    query=vector(candidate)
    if model is None or query is None:return None
    p_raw=float(_sigmoid(np.r_[1.0,query]@model["beta"]))
    p_base=model["base"]
    p_shrunk=0.65*p_raw+0.35*p_base
    expected=p_shrunk*model["win_gain"]+(1-p_shrunk)*model["loss_net"]
    return {"training_completed":model["n"],
            "predicted_win_pct":round(100*p_shrunk,2),
            "baseline_win_pct":round(100*p_base,2),
            "expected_net_pct":round(expected,3),
            "predicted_uplift_pct":round(100*(p_shrunk-p_base),2)}

def fit_probabilities(history,candidate):
    """Backward-compatible per-item scorer (fast policies use date snapshots)."""
    return predict_from_snapshot(prepare_snapshot(history),candidate)

def walkforward_model_filter(trades):
    """One-step chronological prequential scoring with fully exited records."""
    ordered=sorted(trades,key=lambda t:(t["signal_date"],t["ticker"],t["entry_date"]))
    closes=sorted(trades,key=lambda t:(t["exit_date"],t["ticker"],t["entry_date"]))
    accepted=[]
    info={"total":len(trades),"missing_features":0,"insufficient":0,
          "weak_prediction":0,"accepted":0}
    past=[];j=0
    for item in ordered:
        while j<len(closes) and closes[j]["exit_date"]<item["signal_date"]:
            past.append(closes[j]);j+=1
        if vector(item) is None:
            info["missing_features"]+=1
            continue
        prediction=fit_probabilities(past,item)
        if prediction is None:
            info["insufficient"]+=1
            continue
        if (prediction["predicted_win_pct"]/100<MIN_PREDICTED_WIN
            or prediction["predicted_uplift_pct"]/100<MIN_UPLIFT
            or prediction["expected_net_pct"]<MIN_EXPECTED_NET_PCT):
            info["weak_prediction"]+=1
            continue
        accepted.append({**item,**{"model_"+k:v for k,v in prediction.items()}})
        info["accepted"]+=1
    return accepted,info

# Predeclared diagnostic policies. Never select the best one using the
# already-seen 2026 evaluation period.
POLICIES={
    "coverage":{"win_pct":40.0,"uplift_pct":0.0,"expected_net_pct":0.05},
    "balanced":{"win_pct":48.0,"uplift_pct":1.0,"expected_net_pct":0.15},
    "strict":{"win_pct":55.0,"uplift_pct":4.0,"expected_net_pct":0.20},
}

def walkforward_model_policies(trades):
    """One model fit per signal date; scores all trades at that date.

    No candidate outcome can leak into its own or same-day peer's model:
    training history is restricted to exits strictly before signal_date.
    """
    ordered=sorted(trades,key=lambda t:(t["signal_date"],t["ticker"],t["entry_date"]))
    completed=sorted(trades,key=lambda t:(t["exit_date"],t["ticker"],t["entry_date"]))
    selected={key:[] for key in POLICIES}
    status={"total":len(trades),"missing_features":0,"insufficient":0,
            "scored":0,"models_fitted":0,
            "policy_accepted":{key:0 for key in POLICIES}}
    j=0;history=[];active_date=None;model=None
    for item in ordered:
        signal=item["signal_date"]
        if signal!=active_date:
            while j<len(completed) and completed[j]["exit_date"]<signal:
                history.append(completed[j]);j+=1
            model=prepare_snapshot(history)
            active_date=signal
            status["models_fitted"]+=1
        if vector(item) is None:
            status["missing_features"]+=1
            continue
        score=predict_from_snapshot(model,item)
        if score is None:
            status["insufficient"]+=1
            continue
        status["scored"]+=1
        for name,policy in POLICIES.items():
            if (score["predicted_win_pct"]>=policy["win_pct"] and
                score["predicted_uplift_pct"]>=policy["uplift_pct"] and
                score["expected_net_pct"]>=policy["expected_net_pct"]):
                selected[name].append({**item,**{"model_"+k:v for k,v in score.items()}})
                status["policy_accepted"][name]+=1
    return selected,status
