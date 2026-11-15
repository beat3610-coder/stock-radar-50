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

def fit_probabilities(history,candidate):
    """Return prediction at trade decision time, or None for small samples."""
    query=vector(candidate)
    if query is None:return None
    usable=[(vector(t),float(t["net_pct"])) for t in history if vector(t) is not None
            and np.isfinite(float(t.get("net_pct",float("nan"))))]
    if len(usable)<MIN_COMPLETED:return None
    usable=usable[-MAX_TRAIN:]
    x=np.asarray([v for v,_ in usable])
    pnl=np.asarray([r for _,r in usable])
    y=(pnl>0).astype(float)
    n=len(y)
    # All-positive or all-negative samples are not fit; use honest prior only.
    if np.unique(y).size<2:return None
    X=np.column_stack([np.ones(n),x])
    beta=np.zeros(X.shape[1],dtype=float)
    # L2 shrinks noisy indicator coefficients. Intercept gets no penalty.
    penalty=np.diag([0.0]+[16.0]*x.shape[1])
    for _ in range(12):
        p=_sigmoid(X@beta)
        weights=np.maximum(p*(1-p),0.02)
        grad=X.T@(y-p)-penalty@beta
        hessian=X.T@(weights[:,None]*X)+penalty+np.eye(X.shape[1])*1e-7
        delta=np.linalg.solve(hessian,grad)
        beta+=np.clip(delta,-1.5,1.5)
        if np.max(np.abs(delta))<1e-5:break
    p_raw=float(_sigmoid(np.r_[1.0,query]@beta))
    p_base=float((y.sum()+2)/(n+4))
    # Baseline shrink prevents extrapolation producing spurious 90% forecasts.
    p_shrunk=0.65*p_raw+0.35*p_base
    win_magnitude=float(np.mean(pnl[y>0]))
    loss_magnitude=float(np.mean(pnl[y==0]))  # already negative, net of costs
    expected=p_shrunk*win_magnitude+(1-p_shrunk)*loss_magnitude
    return {"training_completed":n,"predicted_win_pct":round(100*p_shrunk,2),
            "baseline_win_pct":round(100*p_base,2),
            "expected_net_pct":round(expected,3),
            "predicted_uplift_pct":round(100*(p_shrunk-p_base),2)}

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
